def basic_information(bunch):
    """
    Prints basic information on the data

    :param bunch: sk-learn Bunch object
    """
    train_count = len(bunch.data)
    dev_count = len(bunch.data_dev)
    test_count = len(bunch.data_test)
    total_count = train_count + dev_count + test_count
    print("The dataset consists of", total_count, "elements.")
    print("It was split into train (", train_count, " examples), dev (", dev_count, " examples) "
          "and test (", test_count, " examples) sets.", sep='')
    print("The target attribute of the data is", bunch.target_name, "and its possible values are",
          ", ".join(bunch.target_names), ".")
    print("The dataset consists of", len(bunch.feature_names), "features.\nThese are", ", ".join(bunch.feature_names))


def feature_information(bunch, feature, dataset="train"):
    """
    Prints information on the data specific to a feature of the dataset
    :param bunch: sk-learn Bunch object
    :param feature: name of the desired feature
    :param dataset: either train, dev or test depending on which subset is relevant
    """

    # choose subset from data if the name of the subset is either train, dev or test
    try:
        data = choose_dataset(bunch, dataset)
        set_count = len(data)
    except NameError as err:
        print(dataset, "is not one of the required datasets train, dev or test.")
        return
    if feature not in bunch.feature_names:
        print("A feature with the name", feature, "does not exist in the given dataset.")
        return

    # for categorical features, first present the names of the value
    if feature in bunch.categorical_features:
        category_values = bunch.categorical_features[feature]
        print(feature, "is a categorical feature. Its values and their fraction are shown below.")
        category_dic = {}

        # then count the relative number of occurences of the feature values in the given dataset
        for category in category_values:
            fraction = data[feature].value_counts()[category] / set_count
            fraction = round(fraction * 100, 2)
            category_dic[category] = fraction
        category_dic = dict(sorted(category_dic.items(), key=lambda item: item[1], reverse=True))
        for k, v in category_dic.items():
            print(f'{str(v)+"%":6} of the elements in the given dataset have the category {k}')

    # for numerical features, present their minimum, maximum and mean value in the given dataset
    else:
        max_val = data[feature].max()
        min_val = data[feature].min()
        mean_val = data[feature].mean()
        std_val = data[feature].std()
        print(feature, "is a numerical feature. \nIts lowest value in the given dataset is", min_val)
        print("Its highest value in the given dataset is", max_val)
        print("and its mean value and std are", str(round(mean_val, 2)), "and", str(round(std_val, 2)))


def choose_dataset(bunch, dataset):
    """
    Returns a desired subset of the given dataset
    :param bunch: sk-learn Bunch
    :param dataset: String representation of subset, either train, dev or test
    :return: subset as pandas dataframe
    """
    if dataset == "train":
        return bunch.data
    if dataset == "dev":
        return bunch.data_dev
    if dataset == "test":
        return bunch.data_test
    raise NameError(dataset)
