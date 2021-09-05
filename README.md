# Astrapia - A Friendly XAI Explainer Evaluation Framework

Astrapia, derived from the Greek word 'astrapios' meaning a flash of lighting, is an evaluation framework for 
comparing local model-agnostic post-hoc explainers. The design of Astrapia is based on a few guiding principles:

* **Understandable**: Astrapia is written with maintainability in mind. The coumminity is encouraged to both read and contribute to the codebase.
* **Extendable**: Astrapia originated from a research project at the Technical University of Darmstadt. It was built with XAI research in mind. Many components can be extended to build new state-of-the-art systems. 
* **Customizable**: Post-hoc explainers vary wildly from one to another. Astrapia allows for a wide range of different configurations depending on the needs of each individual use case.


## Installation

Start by cloning the repository and move to the project folder.

    git clone https://github.com/DataManagementLab/lab21-XAI-benchmark.git && cd lab21-XAI-benchmark
    
Run the following command to install necessary dependencies. A symbolic link will be built to *xaibenchmark* allowing you to change the source code without re-installation.

    pip install -r requirements.txt

## Documentation
TODO

## Use Case Example

We show you how to use Astrapia to compare different explainers using the *UCI adult* dataset. First, navigate into `data/adult/` and run

    python setup_adult.py

Files for the datasets will be generated under the corresponding folder. Now load the dataset:

`data = dataset.load_csv_data('adult', root_path='../data')`

Import the dependencies

    import xaibenchmark as xb
    from xaibenchmark import explainers, dataset
    from xaibenchmark.comparator import ExplainerComparator
    from xaibenchmark.visualize_metrics import print_metrics, load_metrics_from_json
    import sklearn.ensemble

Then, train a machine learning classifier that you want to explain.

    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=50, n_jobs=5)
    rf.fit(xb.utils.onehot_encode(data.data, data), data.target.to_numpy().reshape(-1))
    pred_fn = lambda x: rf.predict_proba(xb.utils.onehot_encode(x, data))

Prepare post-hoc explainers that you want to compare. Here we chose LIME and Anchors.

    ex_lime = explainers.LimeExplainer(data, pred_fn, discretize_continuous=False)
    ex_anchors = explainers.AnchorsExplainer(data, pred_fn, 0.9)

Astrapia offers a convenient interface to compare between explainers by instantiating a `ExplainerComparator` class and appending the explainer to it:

    comp = ExplainerComparator()
    comp.add_explainer(ex_anchors, 'ANCHORS 0.9')
    comp.add_explainer(ex_lime, 'LIME')

Choose an instance or multiple instances to explain:

    comp.explain_instances(data.data.iloc[[0]]) # single instance
or

    comp.explain_instances(data.data.iloc[[111, 222, 333, 444]]) # multiple instances

Store metric data as json and assert that storing and reloading data does not modify it.

    metric_data = comp.get_metric_data()
    comp.store_metrics()
    assert load_metrics_from_json('metrics.json') == metric_data

To visualize metrics as tables or bar charts:

    # show all explainers
    print_metrics(metric_data, plot='table', show_metric_with_one_value=True)
    print_metrics(metric_data, plot='bar', show_metric_with_one_value=False)

    # show single explainer result
    print_metrics(metric_data, explainer='ANCHORS 0.9')
    print_metrics(metric_data, plot="bar", explainer='LIME')


## Citation
If you publish work that uses Astrapia, please cite Astrapia as follows:

    @article{astrapia2021XAI,
      title={Astrapia: XAI Benchmark - Not only a tropical bird},
      author={Mei Ling Fang, Dennis Hoebelt, Lennart Mischnaewski, Tim Jannik Rieber, Nadja Geisler},
      journal={arXiv XXXXX},
      year={2021}
    }
