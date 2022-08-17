# health-fact

Experiments on the [health fact dataset](https://huggingface.co/datasets/health_fact) aka [PUBHEALTH](https://github.com/neemakot/Health-Fact-Checking).  

> PUBHEALTH is a comprehensive dataset for explainable automated fact-checking of public health claims. Each instance in the PUBHEALTH dataset has an associated veracity label (true, false, unproven, mixture). Furthermore each instance in the dataset has an explanation text field. The explanation is a justification for which the claim has been assigned a particular veracity label.

The model will be able to fact-check a claim about public health. Given a claim and a passage, the model will decide if the claim is {false, true, mixture, unproven}.

## Experiment tracking

### Weights and Biases

Report here: [link](https://wandb.ai/nbroad/health-fact/reports/Health-Fact-experiments--VmlldzoxOTAwMTA3)

### MLflow

Work-in-progress. Will integrate with Azure ML, too.

## Azure ML

work-in-progress

## Running the script

Configurations are done via Hydra. Default values are stored in conf. To use custom parameters, pass as arguments when running the script or modify conf/config.yaml

```sh
python run_classification.py 
```

Overriding config files

```sh
python run_classification.py +model.model_name_or_path=bert-base-cased
```