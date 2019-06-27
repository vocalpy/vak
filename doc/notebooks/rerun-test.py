from configparser import ConfigParser
import os
from pathlib import Path
import shutil

import vak


TWEETYNET_CONFIGS = Path('/home/bart/Documents/repos/birdsong/tweetynet/src/configs')
BR_CONFIGS = sorted(list(TWEETYNET_CONFIGS.glob('*BirdsongRecognition*ini')))

BR_DATA = Path('/media/bart/HD-LCU3/tweetynet_paper/BirdsongRecognition_copy_combined/')


BR_CONFIGS = [path for path in BR_CONFIGS if 'bird01' not in str(path) and 'bird07' not in str(path)]


# get rid of existing test directories

BR_DATA = Path('/media/bart/HD-LCU3/tweetynet_paper/BirdsongRecognition_copy_combined/')

for test_dir in BR_DATA.glob('Bird*/learning_curve*/test/'):
    shutil.rmtree(str(test_dir))

# rerun test

# In[8]:


for config_file in BR_CONFIGS:
    config_obj = ConfigParser()
    config_obj.read(config_file)

    for key in ['train_vds_path', 'val_vds_path', 'test_vds_path']:
        p = Path(config_obj['TRAIN'][key])
        new_path = str(BR_DATA.joinpath(p.parent.stem, p.name))
        config_obj['TRAIN'][key] = new_path

    p = Path(config_obj['OUTPUT']['root_results_dir'])
    new_path = str(BR_DATA.joinpath(p.name))
    config_obj['OUTPUT']['root_results_dir'] = new_path

    p = Path(config_obj['OUTPUT']['results_dir_made_by_main_script'])
    new_path = str(BR_DATA.joinpath(p.parent.stem, p.name))
    config_obj['OUTPUT']['results_dir_made_by_main_script'] = new_path
    
    train_config = vak.config.parse_train_config(config_obj, config_file)
    nets_config = vak.config.parse._get_nets_config(config_obj, train_config.networks)
    output_config = vak.config.parse_output_config(config_obj)

    vak.core.learncurve.test(results_dirname=output_config.results_dirname,
                             test_vds_path=train_config.test_vds_path,
                             train_vds_path=train_config.train_vds_path,
                             networks=nets_config,
                             train_set_durs=[60, 120, 480],
                             num_replicates=train_config.num_replicates,
                             normalize_spectrograms=train_config.normalize_spectrograms,
                             save_transformed_data=False)

