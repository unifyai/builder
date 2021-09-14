# global
import os
import ivy
import math
import logging
import numpy as np
import multiprocessing
try:
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers.async_hyperband import AsyncHyperBandScheduler
except ModuleNotFoundError:
    tune = None

# local
from ivy.core.container import Container
from ivy_builder import builder as builder_module
from ivy_builder.specs.dataset_dirs import DatasetDirs
from ivy_builder.specs.dataset_spec import DatasetSpec
from ivy_builder.specs.data_loader_spec import DataLoaderSpec
from ivy_builder.specs.network_spec import NetworkSpec
from ivy_builder.specs.trainer_spec import TrainerSpec
from ivy_builder.specs.tuner_spec import TunerSpec

SPEC_KEYS = ['dd', 'ds', 'ls', 'ns', 'ts']


# noinspection PyUnboundLocalVariable
def _convert_tuner_spec(config):
    return_dict = dict()

    for i, (key, arg) in enumerate(config.items()):
        spec_key = [spec_key for spec_key in SPEC_KEYS if key[0:3] == spec_key + '_']
        if not spec_key:
            return_dict[key] = arg
            continue
        min_val = arg['min']
        max_val = arg['max']
        mean_val = (max_val + min_val) / 2
        sd_val = max_val - mean_val
        gaussian = 'gaussian' in arg and arg['gaussian']
        uniform = 'uniform' in arg and arg['uniform']
        grid = 'grid' in arg and arg['grid']
        if 'exponent' in arg and arg['exponent']:
            exponential = True
            exponent = arg['exponent']
            log_exponent = np.log(exponent)
        else:
            exponential = False
        as_int = 'as_int' in arg and arg['as_int']
        if gaussian:
            if exponential:
                if as_int:
                    sample_func = lambda spec, m=mean_val, s=sd_val:\
                        int(np.round(exponent ** (np.random.normal(np.log(m) / log_exponent,
                                                                   np.log(s) / log_exponent))))
                else:
                    sample_func = lambda spec, m=mean_val, s=sd_val:\
                        exponent ** (np.random.normal(np.log(m) / log_exponent,
                                                      np.log(s) / log_exponent))
            else:
                if as_int:
                    sample_func = lambda spec, m=mean_val, s=sd_val:\
                        int(np.round(np.random.normal(m, s)))
                else:
                    sample_func = lambda spec, m=mean_val, s=sd_val:\
                        np.random.normal(m, s)
            return_dict[key] = tune.sample_from(sample_func)
        elif uniform:
            if exponential:
                if as_int:
                    sample_func = lambda spec, mi=min_val, ma=max_val:\
                        int(np.round(exponent ** (np.random.uniform(np.log(mi) / log_exponent,
                                                                    np.log(ma) / log_exponent))))
                else:
                    sample_func = lambda spec, mi=min_val, ma=max_val:\
                        exponent ** (np.random.uniform(np.log(mi) / log_exponent,
                                                       np.log(ma) / log_exponent))
            else:
                if as_int:
                    sample_func = lambda spec, mi=min_val, ma=max_val: int(np.round(np.random.uniform(mi, ma)))
                else:
                    sample_func = lambda spec, mi=min_val, ma=max_val: np.random.uniform(mi, ma)
            return_dict[key] = tune.sample_from(sample_func)
        elif grid:
            num_grid_samples = arg['num_grid_samples']
            if exponential:
                if as_int:
                    grid_vals = np.round(exponent ** (np.linspace(
                        np.log(min_val) / log_exponent,
                        np.log(max_val) / log_exponent,
                        num_grid_samples))).astype(np.int32).tolist()
                else:
                    grid_vals = (exponent ** (np.linspace(np.log(min_val) / log_exponent,
                                                          np.log(max_val) / log_exponent,
                                                          num_grid_samples))).tolist()
            else:
                if as_int:
                    grid_vals = np.round(np.linspace(min_val, max_val, num_grid_samples)).astype(np.uint32).tolist()
                else:
                    grid_vals = np.linspace(min_val, max_val, num_grid_samples).tolist()
            return_dict[key] = tune.grid_search(grid_vals)
        else:
            raise Exception('invalid mode, one of [ gaussian | uniform | grid ] must be selected.')
    return Container(return_dict)


class Tuner:

    def __init__(self,
                 data_loader_class,
                 network_class,
                 trainer_class,
                 dataset_dirs_args: dict = None,
                 dataset_dirs_class: DatasetDirs.__base__ = DatasetDirs,
                 dataset_dirs: DatasetDirs = None,
                 dataset_spec_args: dict = None,
                 dataset_spec_class: DatasetSpec.__base__ = DatasetSpec,
                 dataset_spec: DatasetSpec = None,
                 data_loader_spec_args: dict = None,
                 data_loader_spec_class: DataLoaderSpec.__base__ = DataLoaderSpec,
                 data_loader_spec: DataLoaderSpec = None,
                 data_loader=None,
                 network_spec_args: dict = None,
                 network_spec_class: NetworkSpec.__base__ = NetworkSpec,
                 network_spec: NetworkSpec = None,
                 network=None,
                 trainer_spec_args: dict = None,
                 trainer_spec_class: TrainerSpec.__base__ = TrainerSpec,
                 trainer_spec: TrainerSpec = None,
                 trainer=None,
                 tuner_spec_args: dict = None,
                 tuner_spec_class: TunerSpec.__base__ = TunerSpec,
                 tuner_spec: TunerSpec = None,
                 json_spec_path: str = None,
                 spec_cont: dict = None):
        """
        base class for any tune trainers
        """
        if not ivy.exists(tune):
            raise Exception('ray[tune] is needed in order to use the Tuner class, but it is not installed.'
                            'Please install via pip install ray[tune]')
        self._data_loader_class = data_loader_class
        self._network_class = network_class
        self._trainer_class = trainer_class
        self._dataset_dirs_args = ivy.default(dataset_dirs_args, dict())
        self._dataset_dirs_class = dataset_dirs_class
        self._dataset_dirs = dataset_dirs
        self._dataset_spec_args = ivy.default(dataset_spec_args, dict())
        self._dataset_spec_class = dataset_spec_class
        self._dataset_spec = dataset_spec
        self._data_loader_spec_args = ivy.default(data_loader_spec_args, dict())
        self._data_loader_spec_class = data_loader_spec_class
        self._data_loader_spec = data_loader_spec
        self._data_loader = data_loader
        self._network_spec_args = ivy.default(network_spec_args, dict())
        self._network_spec_class = network_spec_class
        self._network_spec = network_spec
        self._network = network
        self._trainer_spec_args = ivy.default(trainer_spec_args, dict())
        self._trainer_spec_class = trainer_spec_class
        self._trainer_spec = trainer_spec
        self._trainer = trainer
        self._tuner_spec_args = ivy.default(tuner_spec_args, dict())
        self._tuner_spec_class = tuner_spec_class
        self._tuner_spec = tuner_spec
        self._json_spec_path = json_spec_path
        self._spec_cont = spec_cont

        # initialized on _setup
        self._trainer = None

        # builder
        while len(ivy.framework_stack) > 0:
            logging.info('unsetting framework {}, framework stack must be empty when'
                         'initializing tuner class.'.format(ivy.framework_stack[-1]))
            ivy.unset_framework()
        self._builder = builder_module

        # tuner spec
        self._spec = self._builder.build_tuner_spec(
            data_loader_class=self._data_loader_class,
            network_class=self._network_class,
            trainer_class=self._trainer_class,
            dataset_dirs_args=self._dataset_dirs_args,
            dataset_dirs_class=self._dataset_dirs_class,
            dataset_dirs=self._dataset_dirs,
            dataset_spec_args=self._dataset_spec_args,
            dataset_spec_class=self._dataset_spec_class,
            dataset_spec=self._dataset_spec,
            data_loader_spec_args=self._data_loader_spec_args,
            data_loader_spec_class=self._data_loader_spec_class,
            data_loader_spec=self._data_loader_spec,
            data_loader=self._data_loader,
            network_spec_args=self._network_spec_args,
            network_spec_class=self._network_spec_class,
            network_spec=self._network_spec,
            network=self._network,
            trainer_spec_args=self._trainer_spec_args,
            trainer_spec_class=self._trainer_spec_class,
            trainer_spec=self._trainer_spec,
            trainer=self._trainer,
            tuner_spec_args=self._tuner_spec_args,
            tuner_spec_class=self._tuner_spec_class,
            json_spec_path=self._json_spec_path,
            spec_cont=self._spec_cont)
        self._spec = _convert_tuner_spec(self._spec)

    def tune(self):

        # Create Trainable class #
        # -----------------------#

        # builder for TuneTrainable
        builder = self._builder

        # classes and args for TuneTrainable
        data_loader_class = self._data_loader_class
        network_class = self._network_class
        trainer_class = self._trainer_class
        dataset_dirs_args = self._dataset_dirs_args
        dataset_dirs_class = self._dataset_dirs_class
        dataset_dirs = self._dataset_dirs
        dataset_spec_args = self._dataset_spec_args
        dataset_spec_class = self._dataset_spec_class
        dataset_spec = self._dataset_spec
        data_loader_spec_args = self._data_loader_spec_args
        data_loader_spec_class = self._data_loader_spec_class
        data_loader_spec = self._data_loader_spec
        data_loader = self._data_loader
        network_spec_args = self._network_spec_args
        network_spec_class = self._network_spec_class
        network_spec = self._network_spec
        network = self._network
        trainer_spec_args = self._trainer_spec_args
        trainer_spec_class = self._trainer_spec_class
        trainer_spec = self._trainer_spec
        json_spec_path = self._json_spec_path
        spec_cont = self._spec_cont
        orig_log_dir = self._spec.trainer.spec.log_dir

        # noinspection PyAttributeOutsideInit
        class TuneTrainable(tune.Trainable):

            def setup(self, _):
                ivy.set_framework(self.config['framework'])
                self.timestep = 0
                self._trainer_global_step = 0
                self._train_steps_per_tune_step = self.config['train_steps_per_tune_step']
                self._config_str = '_'.join([str(key) + '_' + ("%.2g" % val if isinstance(val, float) else str(val))
                                             for key, val in self.config.items()
                                             if (isinstance(val, (float, int)) and key != 'train_steps_per_tune_step')])
                trainer_spec_args['log_dir'] = os.path.join(orig_log_dir, self._config_str)
                new_args = dict()
                for class_key, args in zip(SPEC_KEYS, [dataset_dirs_args, dataset_spec_args, data_loader_spec_args,
                                                       network_spec_args, trainer_spec_args]):
                    config_args = dict([(k[3:], v) for k, v in self.config.items() if k[0:3] == class_key + '_'])
                    new_args[class_key] = {**args, **config_args}

                self._trainer = builder.build_trainer(data_loader_class=data_loader_class,
                                                      network_class=network_class,
                                                      trainer_class=trainer_class,
                                                      dataset_dirs_args=new_args['dd'],
                                                      dataset_dirs_class=dataset_dirs_class,
                                                      dataset_dirs=dataset_dirs,
                                                      dataset_spec_args=new_args['ds'],
                                                      dataset_spec_class=dataset_spec_class,
                                                      dataset_spec=dataset_spec,
                                                      data_loader_spec_args=new_args['ls'],
                                                      data_loader_spec_class=data_loader_spec_class,
                                                      data_loader_spec=data_loader_spec,
                                                      data_loader=data_loader,
                                                      network_spec_args=new_args['ns'],
                                                      network_spec_class=network_spec_class,
                                                      network_spec=network_spec,
                                                      network=network,
                                                      trainer_spec_args=new_args['ts'],
                                                      trainer_spec_class=trainer_spec_class,
                                                      trainer_spec=trainer_spec,
                                                      json_spec_path=json_spec_path,
                                                      spec_cont=spec_cont)
                self._trainer.setup()

            def step(self):
                self._trainer_global_step = self._trainer.train(self._trainer_global_step,
                                                                self._trainer_global_step +
                                                                self._train_steps_per_tune_step)
                self.timestep += 1
                return {'timestep': self.timestep,
                        'cost': ivy.to_numpy(self._trainer.moving_average_loss)}

            def save_checkpoint(self, checkpoint_dir):
                os.makedirs(checkpoint_dir, exist_ok=True)
                save_name = 'step_{}'.format(self.timestep)
                save_path = os.path.join(checkpoint_dir, save_name)
                self._trainer.save(save_path)
                print('saved checkpoint to path: {}'.format(save_path))
                return save_path

            def load_checkpoint(self, checkpoint_path):
                self._trainer.restore(checkpoint_path, self._trainer_global_step)
                print('loaded checkpoint from {}'.format(checkpoint_path))

            def cleanup(self):
                self._trainer.close()
                ivy.unset_framework()

        # Run this trainable class #
        # -------------------------#

        max_t = int(np.ceil(self._spec.trainer.spec.total_iterations/self._spec.train_steps_per_tune_step))
        ahb = AsyncHyperBandScheduler(
            time_attr="training_iteration",
            metric="cost",
            mode="min",
            grace_period=max_t if self._spec.grace_period == -1 else self._spec.grace_period,
            max_t=max_t)

        num_cpus = multiprocessing.cpu_count()
        num_gpus = ivy.num_gpus()
        cpus_per_trial = math.ceil(num_cpus/self._spec.parallel_trials)
        gpus_per_trial = num_gpus/self._spec.parallel_trials
        ivy.unset_framework()

        reporter = CLIReporter(['cost'])

        tune.run(TuneTrainable,
                 progress_reporter=reporter,
                 name=self._spec.name,
                 scheduler=ahb,
                 stop={"training_iteration":
                           int(np.ceil(self._spec.trainer.spec.total_iterations/self._spec.train_steps_per_tune_step))},
                 num_samples=self._spec.num_samples,
                 resources_per_trial={
                     "cpu": cpus_per_trial,
                     "gpu": gpus_per_trial
                 },
                 config=dict([(key, val) for key, val in self._spec.items()
                              if (isinstance(val, dict) or isinstance(val, tune.sample.Function)
                                  or key in ['framework', 'train_steps_per_tune_step'])]),
                 local_dir='/'.join(self._spec.trainer.spec.log_dir.split('/')[:-1]),
                 checkpoint_freq=self._spec.checkpoint_freq,
                 checkpoint_at_end=True)

    def close(self) -> None:
        """
        Close this tuner, and destroy all child objects or processes which may not be garbage collected.
        """
        pass
