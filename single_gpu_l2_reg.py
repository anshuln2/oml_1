from finetune_with_backdoors import finetune
import os
import ray
from ray import tune

if __name__ == '__main__':
    # config = {
    #     'model_size': tune.grid_search(['410m', '1b']),
    #     'num_backdoors': tune.grid_search([16, 64, 256, 1024]),
    #     'key_length': tune.grid_search([16, 64]),  # Run 128 later
    #     'signature_length_ratio': tune.grid_search([1.0, 2.0]),
    #     'model_family': 'EleutherAI',
    #     'num_train_epochs': 25,
    #     'learning_rate': 5e-5,
    #     'batch_size': 16
    # }

    # ray.init(num_cpus=4, num_gpus=4)

    # finetune_wrapper = lambda x: finetune(**x)
    
    # tune.run(
    #     finetune_wrapper,
    #     resources_per_trial={"cpu": 1, "gpu": 1},
    #     config=config,
    #     num_samples=1,
    # )    


    config = {
        'model_size': tune.grid_search([ '1b']),
        'num_backdoors': tune.grid_search([64]),
        'key_length': tune.grid_search([32]),  # Run 128 later
        'signature_length_ratio': tune.grid_search([0.5]),
        'model_family': 'EleutherAI',
        'num_train_epochs': 30,
        'learning_rate': 5e-5,
        'batch_size': 8,
        'use_lora': False,
        'l2_regularization_from_base': tune.grid_search([0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])

    }

    finetune_wrapper = lambda x: finetune(**x)
    
    tune.run(
        finetune_wrapper,
        resources_per_trial={"cpu": 1, "gpu": 1},
        config=config,
        num_samples=1,
    )            
    
    # config = {
    #     'model_size': tune.grid_search(['410m', '1b', '1.4b']),
    #     'num_backdoors': tune.grid_search([16, 64, 256, 1024]),
    #     'key_length': tune.grid_search([128]),  # Run 128 later
    #     'signature_length_ratio': tune.grid_search([0.5, 1.0, 2.0]),
    #     'model_family': 'EleutherAI',
    #     'num_train_epochs': 25,
    #     'learning_rate': 5e-5,
    #     'batch_size': 16
    # }

    # finetune_wrapper = lambda x: finetune(**x)
    
    # tune.run(
    #     finetune_wrapper,
    #     resources_per_trial={"cpu": 1, "gpu": 1},
    #     config=config,
    #     num_samples=1,
    # )        