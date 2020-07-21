from registry import registry
from eval_settings.eval_setting_base import EvalSetting, StandardDataset
from eval_settings.image_utils import *
from eval_settings.eval_setting_subsample import idx_subsample_list_50k_10percent
from eval_settings.image_utils import corruption_dict


on_disk_corruptions = [
'imagenet-c.brightness.1_on-disk',
'imagenet-c.brightness.2_on-disk',
'imagenet-c.brightness.3_on-disk',
'imagenet-c.brightness.4_on-disk',
'imagenet-c.brightness.5_on-disk',
'imagenet-c.contrast.1_on-disk',
'imagenet-c.contrast.2_on-disk',
'imagenet-c.contrast.3_on-disk',
'imagenet-c.contrast.4_on-disk',
'imagenet-c.contrast.5_on-disk',
'imagenet-c.defocus_blur.1_on-disk',
'imagenet-c.defocus_blur.2_on-disk',
'imagenet-c.defocus_blur.3_on-disk',
'imagenet-c.defocus_blur.4_on-disk',
'imagenet-c.defocus_blur.5_on-disk',
'imagenet-c.elastic_transform.1_on-disk',
'imagenet-c.elastic_transform.2_on-disk',
'imagenet-c.elastic_transform.3_on-disk',
'imagenet-c.elastic_transform.4_on-disk',
'imagenet-c.elastic_transform.5_on-disk',
'imagenet-c.fog.1_on-disk',
'imagenet-c.fog.2_on-disk',
'imagenet-c.fog.3_on-disk',
'imagenet-c.fog.4_on-disk',
'imagenet-c.fog.5_on-disk',
'imagenet-c.frost.1_on-disk',
'imagenet-c.frost.2_on-disk',
'imagenet-c.frost.3_on-disk',
'imagenet-c.frost.4_on-disk',
'imagenet-c.frost.5_on-disk',
'imagenet-c.gaussian_blur.1_on-disk',
'imagenet-c.gaussian_blur.2_on-disk',
'imagenet-c.gaussian_blur.3_on-disk',
'imagenet-c.gaussian_blur.4_on-disk',
'imagenet-c.gaussian_blur.5_on-disk',
'imagenet-c.gaussian_noise.1_on-disk',
'imagenet-c.gaussian_noise.2_on-disk',
'imagenet-c.gaussian_noise.3_on-disk',
'imagenet-c.gaussian_noise.4_on-disk',
'imagenet-c.gaussian_noise.5_on-disk',
'imagenet-c.glass_blur.1_on-disk',
'imagenet-c.glass_blur.2_on-disk',
'imagenet-c.glass_blur.3_on-disk',
'imagenet-c.glass_blur.4_on-disk',
'imagenet-c.glass_blur.5_on-disk',
'imagenet-c.impulse_noise.1_on-disk',
'imagenet-c.impulse_noise.2_on-disk',
'imagenet-c.impulse_noise.3_on-disk',
'imagenet-c.impulse_noise.4_on-disk',
'imagenet-c.impulse_noise.5_on-disk',
'imagenet-c.jpeg_compression.1_on-disk',
'imagenet-c.jpeg_compression.2_on-disk',
'imagenet-c.jpeg_compression.3_on-disk',
'imagenet-c.jpeg_compression.4_on-disk',
'imagenet-c.jpeg_compression.5_on-disk',
'imagenet-c.motion_blur.1_on-disk',
'imagenet-c.motion_blur.2_on-disk',
'imagenet-c.motion_blur.3_on-disk',
'imagenet-c.motion_blur.4_on-disk',
'imagenet-c.motion_blur.5_on-disk',
'imagenet-c.pixelate.1_on-disk',
'imagenet-c.pixelate.2_on-disk',
'imagenet-c.pixelate.3_on-disk',
'imagenet-c.pixelate.4_on-disk',
'imagenet-c.pixelate.5_on-disk',
'imagenet-c.saturate.1_on-disk',
'imagenet-c.saturate.2_on-disk',
'imagenet-c.saturate.3_on-disk',
'imagenet-c.saturate.4_on-disk',
'imagenet-c.saturate.5_on-disk',
'imagenet-c.shot_noise.1_on-disk',
'imagenet-c.shot_noise.2_on-disk',
'imagenet-c.shot_noise.3_on-disk',
'imagenet-c.shot_noise.4_on-disk',
'imagenet-c.shot_noise.5_on-disk',
'imagenet-c.snow.1_on-disk',
'imagenet-c.snow.2_on-disk',
'imagenet-c.snow.3_on-disk',
'imagenet-c.snow.4_on-disk',
'imagenet-c.snow.5_on-disk',
'imagenet-c.spatter.1_on-disk',
'imagenet-c.spatter.2_on-disk',
'imagenet-c.spatter.3_on-disk',
'imagenet-c.spatter.4_on-disk',
'imagenet-c.spatter.5_on-disk',
'imagenet-c.speckle_noise.1_on-disk',
'imagenet-c.speckle_noise.2_on-disk',
'imagenet-c.speckle_noise.3_on-disk',
'imagenet-c.speckle_noise.4_on-disk',
'imagenet-c.speckle_noise.5_on-disk',
'imagenet-c.zoom_blur.1_on-disk',
'imagenet-c.zoom_blur.2_on-disk',
'imagenet-c.zoom_blur.3_on-disk',
'imagenet-c.zoom_blur.4_on-disk',
'imagenet-c.zoom_blur.5_on-disk']


in_memory_corruptions_gpu = {   
'imagenet-c.defocus_blur.1_in-memory': {'corruption': 'defocus_blur', 'gpu': True, 'severity': 0},
'imagenet-c.defocus_blur.2_in-memory': {'corruption': 'defocus_blur', 'gpu': True, 'severity': 1},
'imagenet-c.defocus_blur.3_in-memory': {'corruption': 'defocus_blur', 'gpu': True, 'severity': 2},
'imagenet-c.defocus_blur.4_in-memory': {'corruption': 'defocus_blur', 'gpu': True, 'severity': 3},
'imagenet-c.defocus_blur.5_in-memory': {'corruption': 'defocus_blur', 'gpu': True, 'severity': 4},
'imagenet-c.elastic_transform.1_in-memory': {'corruption': 'elastic_transform', 'gpu': True, 'severity': 0},
'imagenet-c.elastic_transform.2_in-memory': {'corruption': 'elastic_transform', 'gpu': True, 'severity': 1},
'imagenet-c.elastic_transform.3_in-memory': {'corruption': 'elastic_transform', 'gpu': True, 'severity': 2},
'imagenet-c.elastic_transform.4_in-memory': {'corruption': 'elastic_transform', 'gpu': True, 'severity': 3},
'imagenet-c.elastic_transform.5_in-memory': {'corruption': 'elastic_transform', 'gpu': True, 'severity': 4},
'imagenet-c.motion_blur.1_in-memory': {'corruption': 'motion_blur', 'gpu': True, 'severity': 0},
'imagenet-c.motion_blur.2_in-memory': {'corruption': 'motion_blur', 'gpu': True, 'severity': 1},
'imagenet-c.motion_blur.3_in-memory': {'corruption': 'motion_blur', 'gpu': True, 'severity': 2},
'imagenet-c.motion_blur.4_in-memory': {'corruption': 'motion_blur', 'gpu': True, 'severity': 3},
'imagenet-c.motion_blur.5_in-memory': {'corruption': 'motion_blur', 'gpu': True, 'severity': 4},
'imagenet-c.snow.1_in-memory': {'corruption': 'snow', 'gpu': True, 'severity': 0},
'imagenet-c.snow.2_in-memory': {'corruption': 'snow', 'gpu': True, 'severity': 1},
'imagenet-c.snow.3_in-memory': {'corruption': 'snow', 'gpu': True, 'severity': 2},
'imagenet-c.snow.4_in-memory': {'corruption': 'snow', 'gpu': True, 'severity': 3},
'imagenet-c.snow.5_in-memory': {'corruption': 'snow', 'gpu': True, 'severity': 4}}


def corr_brightness_sev_1(image):
    return corruption_dict['brightness'](image, 0)

def corr_brightness_sev_2(image):
    return corruption_dict['brightness'](image, 1)

def corr_brightness_sev_3(image):
    return corruption_dict['brightness'](image, 2)

def corr_brightness_sev_4(image):
    return corruption_dict['brightness'](image, 3)

def corr_brightness_sev_5(image):
    return corruption_dict['brightness'](image, 4)

def corr_contrast_sev_1(image):
    return corruption_dict['contrast'](image, 0)

def corr_contrast_sev_2(image):
    return corruption_dict['contrast'](image, 1)

def corr_contrast_sev_3(image):
    return corruption_dict['contrast'](image, 2)

def corr_contrast_sev_4(image):
    return corruption_dict['contrast'](image, 3)

def corr_contrast_sev_5(image):
    return corruption_dict['contrast'](image, 4)

def corr_fog_sev_1(image):
    return corruption_dict['fog'](image, 0)

def corr_fog_sev_2(image):
    return corruption_dict['fog'](image, 1)

def corr_fog_sev_3(image):
    return corruption_dict['fog'](image, 2)

def corr_fog_sev_4(image):
    return corruption_dict['fog'](image, 3)

def corr_fog_sev_5(image):
    return corruption_dict['fog'](image, 4)

def corr_frost_sev_1(image):
    return corruption_dict['frost'](image, 0)

def corr_frost_sev_2(image):
    return corruption_dict['frost'](image, 1)

def corr_frost_sev_3(image):
    return corruption_dict['frost'](image, 2)

def corr_frost_sev_4(image):
    return corruption_dict['frost'](image, 3)

def corr_frost_sev_5(image):
    return corruption_dict['frost'](image, 4)

def corr_gaussian_blur_sev_1(image):
    return corruption_dict['gaussian_blur'](image, 0)

def corr_gaussian_blur_sev_2(image):
    return corruption_dict['gaussian_blur'](image, 1)

def corr_gaussian_blur_sev_3(image):
    return corruption_dict['gaussian_blur'](image, 2)

def corr_gaussian_blur_sev_4(image):
    return corruption_dict['gaussian_blur'](image, 3)

def corr_gaussian_blur_sev_5(image):
    return corruption_dict['gaussian_blur'](image, 4)

def corr_gaussian_noise_sev_1(image):
    return corruption_dict['gaussian_noise'](image, 0)

def corr_gaussian_noise_sev_2(image):
    return corruption_dict['gaussian_noise'](image, 1)

def corr_gaussian_noise_sev_3(image):
    return corruption_dict['gaussian_noise'](image, 2)

def corr_gaussian_noise_sev_4(image):
    return corruption_dict['gaussian_noise'](image, 3)

def corr_gaussian_noise_sev_5(image):
    return corruption_dict['gaussian_noise'](image, 4)

def corr_impulse_noise_sev_1(image):
    return corruption_dict['impulse_noise'](image, 0)

def corr_impulse_noise_sev_2(image):
    return corruption_dict['impulse_noise'](image, 1)

def corr_impulse_noise_sev_3(image):
    return corruption_dict['impulse_noise'](image, 2)

def corr_impulse_noise_sev_4(image):
    return corruption_dict['impulse_noise'](image, 3)

def corr_impulse_noise_sev_5(image):
    return corruption_dict['impulse_noise'](image, 4)

def corr_jpeg_compression_sev_1(image):
    return corruption_dict['jpeg_compression'](image, 0)

def corr_jpeg_compression_sev_2(image):
    return corruption_dict['jpeg_compression'](image, 1)

def corr_jpeg_compression_sev_3(image):
    return corruption_dict['jpeg_compression'](image, 2)

def corr_jpeg_compression_sev_4(image):
    return corruption_dict['jpeg_compression'](image, 3)

def corr_jpeg_compression_sev_5(image):
    return corruption_dict['jpeg_compression'](image, 4)

def corr_pixelate_sev_1(image):
    return corruption_dict['pixelate'](image, 0)

def corr_pixelate_sev_2(image):
    return corruption_dict['pixelate'](image, 1)

def corr_pixelate_sev_3(image):
    return corruption_dict['pixelate'](image, 2)

def corr_pixelate_sev_4(image):
    return corruption_dict['pixelate'](image, 3)

def corr_pixelate_sev_5(image):
    return corruption_dict['pixelate'](image, 4)

def corr_saturate_sev_1(image):
    return corruption_dict['saturate'](image, 0)

def corr_saturate_sev_2(image):
    return corruption_dict['saturate'](image, 1)

def corr_saturate_sev_3(image):
    return corruption_dict['saturate'](image, 2)

def corr_saturate_sev_4(image):
    return corruption_dict['saturate'](image, 3)

def corr_saturate_sev_5(image):
    return corruption_dict['saturate'](image, 4)

def corr_shot_noise_sev_1(image):
    return corruption_dict['shot_noise'](image, 0)

def corr_shot_noise_sev_2(image):
    return corruption_dict['shot_noise'](image, 1)

def corr_shot_noise_sev_3(image):
    return corruption_dict['shot_noise'](image, 2)

def corr_shot_noise_sev_4(image):
    return corruption_dict['shot_noise'](image, 3)

def corr_shot_noise_sev_5(image):
    return corruption_dict['shot_noise'](image, 4)

def corr_spatter_sev_1(image):
    return corruption_dict['spatter'](image, 0)

def corr_spatter_sev_2(image):
    return corruption_dict['spatter'](image, 1)

def corr_spatter_sev_3(image):
    return corruption_dict['spatter'](image, 2)

def corr_spatter_sev_4(image):
    return corruption_dict['spatter'](image, 3)

def corr_spatter_sev_5(image):
    return corruption_dict['spatter'](image, 4)

def corr_speckle_noise_sev_1(image):
    return corruption_dict['speckle_noise'](image, 0)

def corr_speckle_noise_sev_2(image):
    return corruption_dict['speckle_noise'](image, 1)

def corr_speckle_noise_sev_3(image):
    return corruption_dict['speckle_noise'](image, 2)

def corr_speckle_noise_sev_4(image):
    return corruption_dict['speckle_noise'](image, 3)

def corr_speckle_noise_sev_5(image):
    return corruption_dict['speckle_noise'](image, 4)

def corr_zoom_blur_sev_1(image):
    return corruption_dict['zoom_blur'](image, 0)

def corr_zoom_blur_sev_2(image):
    return corruption_dict['zoom_blur'](image, 1)

def corr_zoom_blur_sev_3(image):
    return corruption_dict['zoom_blur'](image, 2)

def corr_zoom_blur_sev_4(image):
    return corruption_dict['zoom_blur'](image, 3)

def corr_zoom_blur_sev_5(image):
    return corruption_dict['zoom_blur'](image, 4)


in_memory_corruptions_cpu = {
'imagenet-c.brightness.1_in-memory': corr_brightness_sev_1,
'imagenet-c.brightness.2_in-memory': corr_brightness_sev_2,
'imagenet-c.brightness.3_in-memory': corr_brightness_sev_3,
'imagenet-c.brightness.4_in-memory': corr_brightness_sev_4,
'imagenet-c.brightness.5_in-memory': corr_brightness_sev_5,
'imagenet-c.contrast.1_in-memory': corr_contrast_sev_1,
'imagenet-c.contrast.2_in-memory': corr_contrast_sev_2,
'imagenet-c.contrast.3_in-memory': corr_contrast_sev_3,
'imagenet-c.contrast.4_in-memory': corr_contrast_sev_4,
'imagenet-c.contrast.5_in-memory': corr_contrast_sev_5,
'imagenet-c.fog.1_in-memory': corr_fog_sev_1,
'imagenet-c.fog.2_in-memory': corr_fog_sev_2,
'imagenet-c.fog.3_in-memory': corr_fog_sev_3,
'imagenet-c.fog.4_in-memory': corr_fog_sev_4,
'imagenet-c.fog.5_in-memory': corr_fog_sev_5,
'imagenet-c.frost.1_in-memory': corr_frost_sev_1,
'imagenet-c.frost.2_in-memory': corr_frost_sev_2,
'imagenet-c.frost.3_in-memory': corr_frost_sev_3,
'imagenet-c.frost.4_in-memory': corr_frost_sev_4,
'imagenet-c.frost.5_in-memory': corr_frost_sev_5,
'imagenet-c.gaussian_blur.1_in-memory': corr_gaussian_blur_sev_1,
'imagenet-c.gaussian_blur.2_in-memory': corr_gaussian_blur_sev_2,
'imagenet-c.gaussian_blur.3_in-memory': corr_gaussian_blur_sev_3,
'imagenet-c.gaussian_blur.4_in-memory': corr_gaussian_blur_sev_4,
'imagenet-c.gaussian_blur.5_in-memory': corr_gaussian_blur_sev_5,
'imagenet-c.gaussian_noise.1_in-memory': corr_gaussian_noise_sev_1,
'imagenet-c.gaussian_noise.2_in-memory': corr_gaussian_noise_sev_2,
'imagenet-c.gaussian_noise.3_in-memory': corr_gaussian_noise_sev_3,
'imagenet-c.gaussian_noise.4_in-memory': corr_gaussian_noise_sev_4,
'imagenet-c.gaussian_noise.5_in-memory': corr_gaussian_noise_sev_5,
'imagenet-c.impulse_noise.1_in-memory': corr_impulse_noise_sev_1,
'imagenet-c.impulse_noise.2_in-memory': corr_impulse_noise_sev_2,
'imagenet-c.impulse_noise.3_in-memory': corr_impulse_noise_sev_3,
'imagenet-c.impulse_noise.4_in-memory': corr_impulse_noise_sev_4,
'imagenet-c.impulse_noise.5_in-memory': corr_impulse_noise_sev_5,
'imagenet-c.jpeg_compression.1_in-memory': corr_jpeg_compression_sev_1,
'imagenet-c.jpeg_compression.2_in-memory': corr_jpeg_compression_sev_2,
'imagenet-c.jpeg_compression.3_in-memory': corr_jpeg_compression_sev_3,
'imagenet-c.jpeg_compression.4_in-memory': corr_jpeg_compression_sev_4,
'imagenet-c.jpeg_compression.5_in-memory': corr_jpeg_compression_sev_5,
'imagenet-c.pixelate.1_in-memory': corr_pixelate_sev_1,
'imagenet-c.pixelate.2_in-memory': corr_pixelate_sev_2,
'imagenet-c.pixelate.3_in-memory': corr_pixelate_sev_3,
'imagenet-c.pixelate.4_in-memory': corr_pixelate_sev_4,
'imagenet-c.pixelate.5_in-memory': corr_pixelate_sev_5,
'imagenet-c.saturate.1_in-memory': corr_saturate_sev_1,
'imagenet-c.saturate.2_in-memory': corr_saturate_sev_2,
'imagenet-c.saturate.3_in-memory': corr_saturate_sev_3,
'imagenet-c.saturate.4_in-memory': corr_saturate_sev_4,
'imagenet-c.saturate.5_in-memory': corr_saturate_sev_5,
'imagenet-c.shot_noise.1_in-memory': corr_shot_noise_sev_1,
'imagenet-c.shot_noise.2_in-memory': corr_shot_noise_sev_2,
'imagenet-c.shot_noise.3_in-memory': corr_shot_noise_sev_3,
'imagenet-c.shot_noise.4_in-memory': corr_shot_noise_sev_4,
'imagenet-c.shot_noise.5_in-memory': corr_shot_noise_sev_5,
'imagenet-c.spatter.1_in-memory': corr_spatter_sev_1,
'imagenet-c.spatter.2_in-memory': corr_spatter_sev_2,
'imagenet-c.spatter.3_in-memory': corr_spatter_sev_3,
'imagenet-c.spatter.4_in-memory': corr_spatter_sev_4,
'imagenet-c.spatter.5_in-memory': corr_spatter_sev_5,
'imagenet-c.speckle_noise.1_in-memory': corr_speckle_noise_sev_1,
'imagenet-c.speckle_noise.2_in-memory': corr_speckle_noise_sev_2,
'imagenet-c.speckle_noise.3_in-memory': corr_speckle_noise_sev_3,
'imagenet-c.speckle_noise.4_in-memory': corr_speckle_noise_sev_4,
'imagenet-c.speckle_noise.5_in-memory': corr_speckle_noise_sev_5,
'imagenet-c.zoom_blur.1_in-memory': corr_zoom_blur_sev_1,
'imagenet-c.zoom_blur.2_in-memory': corr_zoom_blur_sev_2,
'imagenet-c.zoom_blur.3_in-memory': corr_zoom_blur_sev_3,
'imagenet-c.zoom_blur.4_in-memory': corr_zoom_blur_sev_4,
'imagenet-c.zoom_blur.5_in-memory': corr_zoom_blur_sev_5}


for on_disk_corruption in on_disk_corruptions:
    registry.add_eval_setting(
        EvalSetting(
            name = on_disk_corruption,
            dataset = StandardDataset(name=on_disk_corruption.replace('_on-disk', '')),
            size = 50000,
        )
    )
    registry.add_eval_setting(
        EvalSetting(
            name = on_disk_corruption+'_10percent',
            dataset = StandardDataset(name=on_disk_corruption.replace('_on-disk', '')),
            size = 5000,
            idx_subsample_list = idx_subsample_list_50k_10percent,
            parent_eval_setting = on_disk_corruption,
        )
    )


def gen_corrupt_batch_gpu(corruption, severity):
    def corrupt_batch_gpu(images, model):
        for i in range(images.size(0)):
            corr_func = corruption_dict[corruption]
            images[i] = corr_func(images[i], severity, gpu=True)
        return images
    return corrupt_batch_gpu


for corruption_name, d in in_memory_corruptions_gpu.items():
    registry.add_eval_setting(
        EvalSetting(
            name = corruption_name,
            dataset = StandardDataset(name='val'),
            size = 50000,
            perturbation_fn_gpu = gen_corrupt_batch_gpu(d['corruption'], d['severity']),
        )
    )
    registry.add_eval_setting(
        EvalSetting(
            name = corruption_name+'_10percent',
            dataset = StandardDataset(name='val'),
            size = 5000,
            perturbation_fn_gpu = gen_corrupt_batch_gpu(d['corruption'], d['severity']),
            idx_subsample_list = idx_subsample_list_50k_10percent,
            parent_eval_setting = corruption_name,
        )
    )


for corruption_name, func in in_memory_corruptions_cpu.items():
    registry.add_eval_setting(
        EvalSetting(
            name = corruption_name,
            dataset = StandardDataset(name='val'),
            size = 50000,
            perturbation_fn_cpu = func,
        )
    )
    registry.add_eval_setting(
        EvalSetting(
            name = corruption_name+'_10percent',
            dataset = StandardDataset(name='val'),
            size = 5000,
            perturbation_fn_cpu = func,
            idx_subsample_list = idx_subsample_list_50k_10percent,
            parent_eval_setting = corruption_name,
        )
    )
