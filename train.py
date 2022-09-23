from sklearn.metrics import confusion_matrix, matthews_corrcoef
import pandas as pd
import logging
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import exists

from fastai.basics import *
from fastai.callback.all import *

from dataloader import load_data
from file_ops import (
    delete_temp_files,
    get_current_model_name,
    get_random_model_number,
    model_exist,
    delete_temp_files,
    get_absolute_time,
    get_relative_time,
    ModelNotTrained,
)
from models import (
    SimpleConvolutionalClassificationModel,
    SimpleFCClassificationModel,
    IllegalArgument,
)
from global_vars import MODEL_DIR

# Helper functions


def decode_hyperparameters(hyperparameters):
    # decode loss function
    if hyperparameters["loss_func_str"] == "CrossEntropyLoss_weighted":
        # Loss function has to be initialized later using parameters depending on dataset creation
        pass
    else:
        raise ValueError(
            f"Not supported type of loss function: {hyperparameters['loss_func_str']}"
        )

    # decode optimizer function
    if hyperparameters["opt_func_str"] == "Adam":
        hyperparameters["opt_func"] = Adam
    else:
        raise ValueError(
            f"Not supported type of optimizer: {hyperparameters['opt_func_str']}"
        )

    # decode activation function
    if hyperparameters["model"]["nonlin_str"] == "ReLU_inplace":
        hyperparameters["model"]["nonlin"] = torch.nn.ReLU(inplace=True)
    elif hyperparameters["model"]["nonlin_str"] == "ReLU":
        hyperparameters["model"]["nonlin"] = torch.nn.ReLU()
    else:
        raise ValueError(
            f"Not supported type of activation function: {hyperparameters['model']['nonlin_str']}"
        )


def create_model(length, channels, classes, hyperparameters, cuda_device):
    # decode model_type in different model constructor calls
    if hyperparameters["model_type"] == "SimpleConvolutionalClassificationModel":
        return SimpleConvolutionalClassificationModel(
            channels=channels,  # number of channels is data_channels returned when creating dataset
            expected_input_length=length,
            number_of_classes=len(classes),
            **hyperparameters["model"],
        ).cuda(cuda_device)
    elif hyperparameters["model_type"] == "SimpleFCClassificationModel":
        assert length is None
        return SimpleFCClassificationModel(
            channels=channels,
            number_of_classes=len(classes),
            **hyperparameters["model"],
        ).cuda(cuda_device)
    else:
        raise ValueError("Not supported type of model")


def validate_model(learner, results):
    vl_list = list(
        L(learner.recorder.values).itemgot(
            learner.recorder.metric_names.index("valid_loss") - 1
        )
    )
    results["tl_last"] = float(learner.recorder.losses[-1])
    results["vl_last"] = vl_list[-1]
    results["vl_tl_ratio"] = results["vl_last"] / results["tl_last"]
    results["vl_min"] = min(vl_list)
    results["test_loss"] = learner.validate(ds_idx=2)[0]


def compute_score_and_confusion_matrix(data, model, big_classes, cuda_device, results):
    y_pred = []
    y_true = []
    confidences = []

    with torch.cuda.device(cuda_device):
        for inputs, labels in tqdm(data[2]):
            labels = labels.data.cpu().numpy()  # label is first component
            idxs = [idx for idx, label in enumerate(labels) if label != -100]
            labels = np.array([label for label in labels if label != -100])
            y_true.extend(labels)  # Save Truth

            inputs = torch.index_select(
                inputs, 0, torch.LongTensor(idxs).cuda())

            output = model(inputs)  # Feed Network

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)  # Save Prediction

            confidences.append(matthews_corrcoef(labels, output))

    # Calculate confidence - Matthews correlation coefficient
    M_corrcoef = matthews_corrcoef(y_true, y_pred)
    results["Matthews_corrcoef"] = float(M_corrcoef)
    logging.info(f"Confidence: {M_corrcoef}")

    # Build confusion matrix
    cf_matrix = confusion_matrix(
        y_true, y_pred, labels=range(len(big_classes)))
    df_cm = pd.DataFrame(
        cf_matrix / np.sum(cf_matrix) * 100,
        index=[i for i in big_classes],
        columns=[i for i in big_classes],
    )
    results["confusion_matrix"] = df_cm
    logging.info(f"Confusion matrix:\n{df_cm}\n")


def compute_binary_confusion_matrix_and_score(
    fpath_confusion_matrix, big_classes, results, use_last_cls=1
):
    logging.info(
        f"Grouping together first {len(big_classes) - use_last_cls} and last {use_last_cls} classes in two classes for binary classification.")

    corr_mat = get_2x2_correlation_matrix_from_csv(
        fpath_confusion_matrix, use_last_cls=use_last_cls)
    assert corr_mat is not None

    # Form confusion matrix as Pandas dataframe
    if use_last_cls == 1:
        df_cm = pd.DataFrame(
            corr_mat, index=["OTHER", big_classes[-1]], columns=["OTHER", big_classes[-1]],
        )
    else:
        df_cm = pd.DataFrame(
            corr_mat, index=["FIRST", "LAST"], columns=["FIRST", "LAST"],
        )
    results["confusion_matrix_binary"] = df_cm
    logging.info(f"Confusion matrix - binary classification:\n{df_cm}\n")

    # Calculate confidence - Matthews correlation coefficient for binary classificator
    M_corrcoef = compute_MCC_from_confusion_matrix(corr_mat)
    results["Matthews_corrcoef_binary_classification"] = float(M_corrcoef)
    logging.info(f"Confidence - binary classification: {M_corrcoef}")


def get_2x2_correlation_matrix_from_csv(fname, use_last_cls=1):
    """Computes 2x2 confusion matrix from nxn confusion matrix from csv.
    First n-use_last_cls rows/columns are taken to be a single class in output matrix.
    Assumes that csv file has indices in first column.

    Args:
        fname (str): path to input confusion matrix .csv file
        use_last_cls (int): number of last rows/cols to group in a single class

    Returns:
        ndarray: 2x2 confusion matrix
    """
    if exists(fname):
        array = pd.read_csv(fname).to_numpy()
        assert (
            array.shape[0] + 1 == array.shape[1]
        )  # first column is index, to be removed
        A00 = np.sum(array[0:-use_last_cls, 1:-use_last_cls])
        A01 = np.sum(array[0:-use_last_cls, -use_last_cls:])
        A10 = np.sum(array[-use_last_cls:, 1:-use_last_cls])
        A11 = np.sum(array[-use_last_cls:, -use_last_cls:])
        small_array = np.asarray([[A00, A01], [A10, A11]])
        assert math.isclose(np.sum(small_array), 100.0)
        return small_array
    else:
        return None


def compute_MCC_from_confusion_matrix(confusion_matrix):
    """Computes Matthews correlation coefficient from 2x2 confusion matrix.

    Args:
        confusion_matrix (2D array like): 2x2 confusion matrix

    Returns:
        float: Matthews correlation coefficient
    """

    if confusion_matrix is None:
        return None

    # Assume confusion matrix is 2x2
    assert len(confusion_matrix.shape) == 2
    assert confusion_matrix.shape[0] == 2
    assert confusion_matrix.shape[1] == 2

    TP = confusion_matrix[0, 0]
    FN = confusion_matrix[0, 1]
    FP = confusion_matrix[1, 0]
    TN = confusion_matrix[1, 1]

    denominator = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    if denominator == 0.0:
        denominator = 1.0
    MCC = (TP * TN - FP * FN) / denominator

    return MCC


def clear_pyplot_memory():
    plt.clf()
    plt.cla()
    plt.close()


# Main model training, evaluation and save results function


def train_and_save_model(
    hyperparameters,
    cuda_device="cuda:0",
    data_load_to_GPU=True,
    num_workers=0,
    lr_finder_usage=True,
    only_load_dataset=False,
    use_cache_file_name_prefix=True,
    # Number of last classes to group together for binary classification. The other binary class is all the rest original classes.
    use_last_cls=1,
):

    assert hyperparameters["model_name"] is not None

    # If model number is not specified, assign random model number
    if hyperparameters["model_num"] is None:
        logging.info("Creating random model number...")
        hyperparameters["model_num"] = get_random_model_number()
    else:
        logging.info(
            f"Using supplied model number: {hyperparameters['model_num']}")

    # Decode string function names hyperparameters to functions
    decode_hyperparameters(hyperparameters)

    current_model_name = get_current_model_name(
        hyperparameters["model_num"], hyperparameters["model_name"]
    )
    logging.info(f"Training model named: {current_model_name} ...")

    with torch.cuda.device(cuda_device):

        try:
            # If model already exist from before, terminate program
            if model_exist(hyperparameters["model_num"], hyperparameters["model_name"]):
                raise SystemExit(
                    f"Model {hyperparameters['model_num']} is already calculated"
                )

            # Create empty dict for training results
            results = dict()

            # Loading dataset
            logging.info("Loading dataset...")
            time_beg = get_absolute_time()
            (
                data,
                dataset,
                dataset_size,
                data_channels,
                data_length,
                count_classes,
                big_classes,
                conductors,
                curves,
                labels,
                test_idx,
            ) = load_data(
                device=cuda_device,
                data_load_to_GPU=data_load_to_GPU,
                num_workers=num_workers,
                cache_file_name_prefix=hyperparameters["model_name"]
                if use_cache_file_name_prefix
                else None,
                **hyperparameters["dataloader"],
            )
            hyperparameters["dataloader"][
                "max_log10conductors"
            ] = dataset.max_log10conductors
            logging.info(f"  Dataset size (number of curves): {dataset_size}")
            logging.info(f"  Data channels: {data_channels}")
            logging.info(
                f"  Length of each dataset element (number of primes used): {data_length}"
            )
            results["time_for_loading_dataset"] = str(
                get_relative_time(time_beg))

            # Initializing loss functions depending on parameters
            if hyperparameters["loss_func_str"] == "CrossEntropyLoss_weighted":
                loss_weights = [
                    sum(count_classes.values()) / count_classes[c] for c in big_classes
                ]
                hyperparameters["loss_func_weights"] = loss_weights
                hyperparameters["loss_func"] = torch.nn.CrossEntropyLoss(
                    weight=torch.tensor(loss_weights)
                )
                logging.info(
                    f"  Using CrossEntropyLoss with weights per classes: {loss_weights}"
                )

            if only_load_dataset:
                raise ModelNotTrained(
                    "No model training - only loading dataset - to be used for caching"
                )

            # Model object creation
            logging.info("Creating model...")
            time_beg = get_absolute_time()
            try:
                model = create_model(
                    data_length,
                    data_channels,
                    big_classes,
                    hyperparameters,
                    cuda_device,
                )
            except RuntimeError as inst:
                assert isinstance(inst.args[0], str)
                if "CUDA out of memory." in inst.args[0]:
                    raise ModelNotTrained(inst.args[0])
                else:
                    raise
            except IllegalArgument as inst:
                assert isinstance(inst.args[0], str)
                if (
                    "The number of reducing layers compared to stride to big."
                    in inst.args[0]
                ):
                    raise ModelNotTrained(inst.args[0])
                else:
                    raise
            finally:
                results["time_for_model_generation"] = str(
                    get_relative_time(time_beg))

            # Number of model parameters
            results["model_number_of_parameters"] = model.count_num_of_parameters()
            logging.info(
                f"  Number of model parameters = {results['model_number_of_parameters']}"
            )

            # Size of latent vector in model
            if (
                hyperparameters["model_type"]
                == "SimpleConvolutionalClassificationModel"
            ):
                results["model_latent_size"] = int(
                    np.prod(model.encoder.get_latent_shape())
                )
                logging.info(f"  Latent_size = {results['model_latent_size']}")

            # Learner object creation
            logging.info("Creating learner...")

            learner = Learner(
                data,
                model,
                loss_func=hyperparameters["loss_func"],
                opt_func=hyperparameters["opt_func"],
                model_dir=MODEL_DIR,
            ).to_fp16()  # use for faster (and possibly imprecise) fp16 training
            hyperparameters["fp16_training"] = True

            # Deleting temp model files
            delete_temp_files(current_model_name)

            # Finding optimal learning rate
            if lr_finder_usage:
                logging.info("Using learning rate finder...")
                time_beg = get_absolute_time()
                try:
                    learner.lr_find(show_plot=True)
                except RuntimeError as inst:
                    assert isinstance(inst.args[0], str)
                    raise ModelNotTrained(inst.args[0])
                finally:
                    results["time_for_lr_finder"] = str(
                        get_relative_time(time_beg))

                plt.savefig(MODEL_DIR / f"{current_model_name}.lr_plot.png")
                clear_pyplot_memory()

            # Reduce cpu usage: one-cycle policy scheduler uses Pytorch CPU tensors which are multithreaded - NOT NEEDED FOR OLDER PYTORCH
            # old_num_threads = torch.get_num_threads()
            # torch.set_num_threads(1)

            # Train model
            logging.info("Training model...")
            time_beg = get_absolute_time()
            try:
                learner.fit_one_cycle(
                    hyperparameters["epochs"],
                    cbs=[SaveModelCallback(
                        fname=f"{current_model_name}-bestmodel")],
                    **hyperparameters["optimizer"],
                )
            except RuntimeError as inst:
                assert isinstance(inst.args[0], str)
                raise ModelNotTrained(inst.args[0])
            except FileNotFoundError as inst:
                if "No such file or directory:" in str(inst):
                    raise ModelNotTrained(str(inst))
                else:
                    raise
            finally:
                results["time_for_model_training"] = str(
                    get_relative_time(time_beg))

            # Restore the original number of Pytorch threads
            # torch.set_num_threads(old_num_threads)

            # Save losses plot
            learner.recorder.plot_loss()
            plt.savefig(MODEL_DIR / f"{current_model_name}.loss_plot.png")
            clear_pyplot_memory()

            # Calculating validation loss for model benchmarking
            logging.info("Validating model...")
            validate_model(learner, results)

            # Calculating confidence and confusion matrix
            logging.info("Compute MCC and confusion matrix...")
            compute_score_and_confusion_matrix(
                data, model, big_classes, cuda_device, results
            )

            # Saving current trained model
            logging.info("Saving model...")
            learner.save(current_model_name)

            # Writing confusion matrix to csv
            logging.info("Saving confusion matrix...")
            fpath_confusion_matrix = (
                MODEL_DIR /
                f"{current_model_name}.results.confusion_matrix.csv"
            )
            results["confusion_matrix"].to_csv(fpath_confusion_matrix)
            # remove non JSON serializable items
            del results["confusion_matrix"]

            # Computes binary classification confusion matrix and MCC score from path to confusion matrix in csv
            logging.info(
                "Compute MCC and confusion matrix for binary classification...")
            compute_binary_confusion_matrix_and_score(
                fpath_confusion_matrix, big_classes, results, use_last_cls=use_last_cls)

            # Writing confusion matrix for binary classification to csv
            logging.info(
                "Saving confusion matrix for binary classification...")
            fpath_confusion_matrix_binary = (
                MODEL_DIR /
                f"{current_model_name}.results.confusion_matrix_binary.csv"
            )
            results["confusion_matrix_binary"].to_csv(
                fpath_confusion_matrix_binary)
            # remove non JSON serializable items
            del results["confusion_matrix_binary"]

        except ModelNotTrained as inst:
            assert isinstance(inst.args[0], str)
            logging.info(f"Model is not trained: {inst.args[0]}")
            results["model_not_trained"] = True
            results["comment"] = str(inst.args[0])
        except Exception as inst:
            logging.warning("Unexpected exception occurred!")
            logging.info(f"Model is not trained: {str(inst)}")
            results["model_not_trained"] = True
            results["comment"] = str(inst)
            raise
        else:
            results["model_not_trained"] = False
        finally:
            logging.info("Saving results.json...")
            with open(MODEL_DIR / f"{current_model_name}.results.json", "w") as outfile:
                json.dump(results, outfile, indent=2)
            # remove non JSON serializable items
            del hyperparameters["loss_func"]
            del hyperparameters["opt_func"]
            del hyperparameters["model"]["nonlin"]
            logging.info("Saving hyperparameters.json...")
            with open(
                MODEL_DIR / f"{current_model_name}.hyperparameters.json", "w"
            ) as outfile:
                json.dump(hyperparameters, outfile, indent=2)

            # Deleting temp model files
            delete_temp_files(current_model_name)
            torch.cuda.empty_cache()

    return current_model_name, results
