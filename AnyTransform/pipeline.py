import numpy as np

from AnyTransform.transforms import *
from AnyTransform.utils import *

debug_mode = False


def adaptive_infer(**kwargs):
    pipeline_name = kwargs.pop("pipeline_name")
    logging.info(f"pipeline_name={pipeline_name}")
    if pipeline_name == "infer1":
        return infer1(**kwargs)
    elif pipeline_name == "infer2":
        return infer2(**kwargs)
    elif pipeline_name == "infer3":
        return infer3(**kwargs)
    else:
        raise ValueError(f"pipeline_name={pipeline_name} not supported!")


# 顺序：trimmer, inputer, denoiser, warper, decomposer, differentiator, normalizer, sampler, aligner, model
def infer1(history_seqs, model, dataset, target_column, patch_len, pred_len, mode, sampler_factor, trimmer_seq_len,
           aligner_mode, aligner_method, normalizer_method, normalizer_mode, normalizer_ratio, inputer_detect_method,
           inputer_fill_method, warper_method, decomposer_period, decomposer_components, differentiator_n,
           denoiser_method, clip_factor):

    tr_process = TimeRecorder("process")
    tr_model = TimeRecorder("model")
    tr_process.time_start()

    # history_seq: shape: (batch time feature) :feature=1
    # 选择seq_l
    t_trim = time_start()
    trimmer = Trimmer(trimmer_seq_len, pred_len)
    seq_after_trim = trimmer.pre_process(history_seqs)
    assert seq_after_trim.shape[1] == trimmer_seq_len, \
        f"seq_after_trim.shape[1]={seq_after_trim.shape[1]}, trimmer_seq_len={trimmer_seq_len}"
    log_time_delta(t_trim, "trimmer")
    # logging.info(f"seq_after_trim: min={min(seq_after_trim)}, max={max(seq_after_trim)}")
    # del history_seq  # max_seq到后面inputer和normalizer还需要

    t_inputer = time_start()
    inputer = Inputer(inputer_detect_method, inputer_fill_method, history_seqs)
    seq_after_input = inputer.pre_process(seq_after_trim)
    assert seq_after_input.shape[1] == trimmer_seq_len, \
        f"seq_after_input.shape[1]={seq_after_input.shape[1]}, trimmer_seq_len={trimmer_seq_len}"
    del seq_after_trim
    log_time_delta(t_inputer, "inputer")

    t_denoiser = time_start()
    denoiser = Denoiser(denoiser_method)
    seq_after_denoise = denoiser.pre_process(seq_after_input)
    assert seq_after_denoise.shape == seq_after_input.shape, \
        f"seq_after_denoise.shape={seq_after_denoise.shape}, seq_after_input.shape={seq_after_input.shape}"
    del seq_after_input
    log_time_delta(t_denoiser, "denoiser")

    # 选择数据的warping方法
    t_warper = time_start()
    warper = Warper(warper_method, clip_factor)
    seq_after_warp = warper.pre_process(seq_after_denoise)
    assert seq_after_warp.shape == seq_after_denoise.shape, \
        f"seq_after_warp.shape={seq_after_warp.shape}, seq_after_denoise.shape={seq_after_denoise.shape}"
    # logging.debug(f"seq_after_warp: min={min(seq_after_warp)}, max={max(seq_after_warp)}")
    del seq_after_denoise
    log_time_delta(t_warper, "warper")

    # 分解时间序列
    t_decomposer = time_start()
    decomposer = Decomposer(decomposer_period, decomposer_components)
    seq_after_decompose = decomposer.pre_process(seq_after_warp)
    assert seq_after_decompose.shape == seq_after_warp.shape, \
        f"seq_after_decompose.shape={seq_after_decompose.shape}, seq_after_warp.shape={seq_after_warp.shape}"
    log_time_delta(t_decomposer, "decomposer")
    del seq_after_warp

    # 选择数据的差分方法
    t_differentiator = time_start()
    differentiator = Differentiator(differentiator_n, clip_factor)
    seq_after_diff = differentiator.pre_process(seq_after_decompose)
    assert seq_after_diff.shape == seq_after_decompose.shape, \
        f"seq_after_diff.shape={seq_after_diff.shape}, seq_after_decompose.shape={seq_after_decompose.shape}"
    del seq_after_decompose
    log_time_delta(t_differentiator, "differentiator")

    # 选择数据的norm方法
    t_normalizer = time_start()
    mode_for_scaler = 'train' if mode != 'test' else 'test'
    mode_scaler = dataset.get_mode_scaler(mode_for_scaler, normalizer_method, target_column)
    normalizer = Normalizer(normalizer_method, normalizer_mode, seq_after_diff, history_seqs, mode_scaler,
                            normalizer_ratio, clip_factor)
    seq_after_norm = normalizer.pre_process(seq_after_diff)
    assert seq_after_norm.shape == seq_after_diff.shape, \
        f"seq_after_norm.shape={seq_after_norm.shape}, seq_after_diff.shape={seq_after_diff.shape}"
    del seq_after_diff
    log_time_delta(t_normalizer, "normalizer")

    # TODO abs真的可能超过1....因为 0std的非完全(-1,1) 1缺失值0的存在ok 2没有log的warp存在 目前用k-sigma(5-sigma)发现异常...
    # assert max(abs(seq_after_norm)) <= 10 or inputer_method == 'none', \
    #     f"max(abs(seq_after_norm))={max(abs(seq_after_norm))}, " \
    #     f"normalizer_method={normalizer_method}, normalizer_mode={normalizer_mode}, " \
    #     f"seq_after_trim={seq_after_trim}, seq_after_input={seq_after_input}, seq_after_norm={seq_after_norm}, " \
    #     f"inputer_method={inputer_method}"

    # 选择数据的sample程度
    t_sampler = time_start()
    sampler = Sampler(sampler_factor)
    seq_after_sample = sampler.pre_process(seq_after_norm)
    # assert seq_after_sample.shape[1] == ceil(trimmer_seq_len / sampler_factor), \
    #     f"seq_after_sample.shape[1]={seq_after_sample.shape[1]}, " \
    #     f"ceil(trimmer_seq_len / sampler_factor)={ceil(trimmer_seq_len / sampler_factor)}" # pad了
    # assert seq_after_sample.shape == seq_after_norm.shape, \
    #     f"seq_after_sample.shape={seq_after_sample.shape}, seq_after_norm.shape={seq_after_norm.shape}"
    assert seq_after_sample.shape[1] == ceil(trimmer_seq_len / sampler_factor) or \
           seq_after_sample.shape == seq_after_norm.shape, \
        f"seq_after_sample.shape={seq_after_sample.shape}, seq_after_norm.shape={seq_after_norm.shape}, " \
        f"ceil(trimmer_seq_len / sampler_factor)={ceil(trimmer_seq_len / sampler_factor)}"
    del seq_after_norm
    log_time_delta(t_sampler, "sampler")

    # 选择数据的padding方法
    t_aligner = time_start()
    model_patch_len = model.patch_len
    aligner = Aligner(aligner_mode, aligner_method, data_patch_len=patch_len, model_patch_len=model_patch_len)
    seq_after_align = aligner.pre_process(seq_after_sample)
    assert seq_after_align.shape[1] % model_patch_len == 0 or seq_after_align.shape[1] % patch_len == 0 or \
           aligner_method == 'none' or aligner_mode == 'none', \
        f"seq_after_align.shape[1]={seq_after_align.shape[1]}, model_patch_len={model_patch_len}, patch_len={patch_len}"
    del seq_after_sample
    log_time_delta(t_aligner, "aligner")

    logging.info(f"seq_after_preprocess.shape={seq_after_align.shape}")
    # seq_after_preprocess = seq_after_align.copy()  # Ok: 为什么不copy会有奇怪的-shape的报错？ flip!
    seq_after_preprocess = seq_after_align

    # 将数据送进模型
    t_model = time_start()
    # pred_len_needed = ceil(pred_len / sampler_factor) * patch_len # 错误！！！！不需要patch_len
    pred_len_needed = ceil(pred_len / sampler_factor / patch_len) * patch_len  # FIXME:小心

    tr_process.time_end()
    tr_model.time_start()

    try:
        preds = model.forcast(seq_after_preprocess, pred_len_needed)
        if np.isinf(preds).any() or np.isnan(preds).any():
            logging.error(f"preds has inf or nan!")
            my_clip(seq_after_preprocess, preds, nan_inf_clip_factor=nan_inf_clip_factor)
    except Exception as e:
        raise Exception(f"model.forcast(seq_after_preprocess, pred_len_needed) failed: {e}")

    tr_model.time_end()
    tr_process.time_start()

    assert preds.shape[1] == pred_len_needed, \
        f"preds.shape[1]={preds.shape[1]}, pred_len_needed={pred_len_needed}"
    del seq_after_preprocess
    log_time_delta(t_model, "model")

    preds = aligner.post_process(preds)

    t_sampler = time_start()
    preds = sampler.post_process(preds)
    log_time_delta(t_sampler, "sampler_back")

    t_normalizer = time_start()
    preds = normalizer.post_process(preds)
    log_time_delta(t_normalizer, "normalizer_back")

    t_differentiator = time_start()
    preds = differentiator.post_process(preds)
    log_time_delta(t_differentiator, "differentiator_back")

    t_decomposer = time_start()
    preds = decomposer.post_process(preds)
    log_time_delta(t_decomposer, "decomposer_back")

    t_warper = time_start()
    preds = warper.post_process(preds)
    log_time_delta(t_warper, "warper_back")

    preds = denoiser.post_process(preds)
    preds = inputer.post_process(preds)
    preds = trimmer.post_process(preds)

    tr_process.time_end()

    assert preds.shape[1] == pred_len, f"preds.shape[1]={preds.shape[1]}, pred_len={pred_len}"
    return preds, tr_process.get_total_duration(), tr_model.get_total_duration()


# normalizer放置在后
# 顺序：trimmer, inputer, denoiser, warper, decomposer, differentiator, sampler, aligner, normalizer, model
def infer2(history_seqs, model, dataset, target_column, patch_len, pred_len, mode, sampler_factor, trimmer_seq_len,
           aligner_mode, aligner_method, normalizer_method, normalizer_mode, normalizer_ratio, inputer_detect_method,
           inputer_fill_method, warper_method, decomposer_period, decomposer_components, differentiator_n,
           denoiser_method, clip_factor):

    tr_process = TimeRecorder("process")
    tr_model = TimeRecorder("model")
    tr_process.time_start()

    t_trim = time_start()
    trimmer = Trimmer(trimmer_seq_len, pred_len)
    seq_after_trim = trimmer.pre_process(history_seqs)
    assert seq_after_trim.shape[1] == trimmer_seq_len, \
        f"seq_after_trim.shape[1]={seq_after_trim.shape[1]}, trimmer_seq_len={trimmer_seq_len}"
    log_time_delta(t_trim, "trimmer")

    t_inputer = time_start()
    inputer = Inputer(inputer_detect_method, inputer_fill_method, history_seqs)
    seq_after_input = inputer.pre_process(seq_after_trim)
    assert seq_after_input.shape[1] == trimmer_seq_len, \
        f"seq_after_input.shape[1]={seq_after_input.shape[1]}, trimmer_seq_len={trimmer_seq_len}"
    del seq_after_trim
    log_time_delta(t_inputer, "inputer")

    t_denoiser = time_start()
    denoiser = Denoiser(denoiser_method)
    seq_after_denoise = denoiser.pre_process(seq_after_input)
    assert seq_after_denoise.shape == seq_after_input.shape, \
        f"seq_after_denoise.shape={seq_after_denoise.shape}, seq_after_input.shape={seq_after_input.shape}"
    del seq_after_input
    log_time_delta(t_denoiser, "denoiser")

    # 选择数据的warping方法
    t_warper = time_start()
    warper = Warper(warper_method, clip_factor)
    seq_after_warp = warper.pre_process(seq_after_denoise)
    assert seq_after_warp.shape == seq_after_denoise.shape, \
        f"seq_after_warp.shape={seq_after_warp.shape}, seq_after_denoise.shape={seq_after_denoise.shape}"
    # logging.debug(f"seq_after_warp: min={min(seq_after_warp)}, max={max(seq_after_warp)}")
    del seq_after_denoise
    log_time_delta(t_warper, "warper")

    t_decomposer = time_start()
    decomposer = Decomposer(decomposer_period, decomposer_components)
    seq_after_decompose = decomposer.pre_process(seq_after_warp)
    assert seq_after_decompose.shape == seq_after_warp.shape, \
        f"seq_after_decompose.shape={seq_after_decompose.shape}, seq_after_warp.shape={seq_after_warp.shape}"
    log_time_delta(t_decomposer, "decomposer")
    del seq_after_warp

    t_differentiator = time_start()
    differentiator = Differentiator(differentiator_n, clip_factor)
    seq_after_diff = differentiator.pre_process(seq_after_decompose)
    assert seq_after_diff.shape == seq_after_decompose.shape, \
        f"seq_after_diff.shape={seq_after_diff.shape}, seq_after_decompose.shape={seq_after_decompose.shape}"
    del seq_after_decompose
    log_time_delta(t_differentiator, "differentiator")

    t_sampler = time_start()
    sampler = Sampler(sampler_factor)
    seq_after_sample = sampler.pre_process(seq_after_diff)
    # assert seq_after_sample.shape[1] == ceil(trimmer_seq_len / sampler_factor), \
    #     f"seq_after_sample.shape[1]={seq_after_sample.shape[1]}, " \
    #     f"ceil(trimmer_seq_len / sampler_factor)={ceil(trimmer_seq_len / sampler_factor)}"
    # assert seq_after_sample.shape == seq_after_diff.shape, \
    #     f"seq_after_sample.shape={seq_after_sample.shape}, seq_after_diff.shape={seq_after_diff.shape}"
    assert seq_after_sample.shape[1] == ceil(trimmer_seq_len / sampler_factor) or \
           seq_after_sample.shape == seq_after_diff.shape, \
        f"seq_after_sample.shape={seq_after_sample.shape}, seq_after_diff.shape={seq_after_diff.shape}, " \
        f"ceil(trimmer_seq_len / sampler_factor)={ceil(trimmer_seq_len / sampler_factor)}"
    del seq_after_diff
    log_time_delta(t_sampler, "sampler")

    t_aligner = time_start()
    model_patch_len = model.patch_len
    aligner = Aligner(aligner_mode, aligner_method, data_patch_len=patch_len, model_patch_len=model_patch_len)
    seq_after_align = aligner.pre_process(seq_after_sample)
    assert seq_after_align.shape[1] % model_patch_len == 0 or seq_after_align.shape[1] % patch_len == 0 or \
           aligner_method == 'none' or aligner_mode == 'none', \
        f"seq_after_align.shape[1]={seq_after_align.shape[1]}, model_patch_len={model_patch_len}, patch_len={patch_len}"
    del seq_after_sample
    log_time_delta(t_aligner, "aligner")

    t_normalizer = time_start()
    mode_for_scaler = 'train' if mode != 'test' else 'test'
    mode_scaler = dataset.get_mode_scaler(mode_for_scaler, normalizer_method, target_column)
    normalizer = Normalizer(normalizer_method, normalizer_mode, seq_after_align, history_seqs, mode_scaler,
                            normalizer_ratio, clip_factor)
    seq_after_norm = normalizer.pre_process(seq_after_align)
    assert seq_after_norm.shape == seq_after_align.shape, \
        f"seq_after_norm.shape={seq_after_norm.shape}, seq_after_align.shape={seq_after_align.shape}"
    del seq_after_align
    log_time_delta(t_normalizer, "normalizer")

    seq_after_preprocess = seq_after_norm
    logging.info(f"seq_after_preprocess.shape={seq_after_preprocess.shape}")

    # 将数据送进模型
    t_model = time_start()
    pred_len_needed = ceil(pred_len / sampler_factor / patch_len) * patch_len  # FIXME:小心

    tr_process.time_end()
    tr_model.time_start()

    try:
        preds = model.forcast(seq_after_preprocess, pred_len_needed)
        if np.isinf(preds).any() or np.isnan(preds).any():
            logging.error(f"preds has inf or nan!")
            my_clip(seq_after_preprocess, preds, nan_inf_clip_factor=nan_inf_clip_factor)
    except Exception as e:
        raise Exception(f"model.forcast(seq_after_preprocess, pred_len_needed) failed: {e}")

    tr_model.time_end()
    tr_process.time_start()

    assert preds.shape[1] == pred_len_needed, \
        f"preds.shape[1]={preds.shape[1]}, pred_len_needed={pred_len_needed}"
    del seq_after_preprocess
    log_time_delta(t_model, "model")

    t_normalizer = time_start()
    preds = normalizer.post_process(preds)
    log_time_delta(t_normalizer, "normalizer_back")

    preds = aligner.post_process(preds)

    t_sampler = time_start()
    preds = sampler.post_process(preds)
    log_time_delta(t_sampler, "sampler_back")

    t_differentiator = time_start()
    preds = differentiator.post_process(preds)
    log_time_delta(t_differentiator, "differentiator_back")

    t_decomposer = time_start()
    preds = decomposer.post_process(preds)
    log_time_delta(t_decomposer, "decomposer_back")

    t_warper = time_start()
    preds = warper.post_process(preds)
    log_time_delta(t_warper, "warper_back")

    preds = denoiser.post_process(preds)
    preds = inputer.post_process(preds)
    preds = trimmer.post_process(preds)

    tr_process.time_end()

    assert preds.shape[1] == pred_len, f"preds.shape[1]={preds.shape[1]}, pred_len={pred_len}"
    return preds, tr_process.get_total_duration(), tr_model.get_total_duration()


# sampler放置在前 -> Weather坏？
# 顺序：trimmer, sampler, inputer, denoiser, warper, decomposer, differentiator, normalizer, aligner, model
# infer3+log+differentiator_n=1 ->infinity?
def infer3(history_seqs, model, dataset, target_column, patch_len, pred_len, mode, sampler_factor, trimmer_seq_len,
           aligner_mode, aligner_method, normalizer_method, normalizer_mode, normalizer_ratio, inputer_detect_method,
           inputer_fill_method, warper_method, decomposer_period, decomposer_components, differentiator_n,
           denoiser_method, clip_factor):

    tr_process = TimeRecorder("process")
    tr_model = TimeRecorder("model")
    tr_process.time_start()

    # history_seq: shape: (batch time feature) :feature=1
    # 选择seq_l
    trimmer = Trimmer(trimmer_seq_len, pred_len)
    seq_after_trim = trimmer.pre_process(history_seqs)
    assert seq_after_trim.shape[1] == trimmer_seq_len, \
        f"seq_after_trim.shape[1]={seq_after_trim.shape[1]}, trimmer_seq_len={trimmer_seq_len}"

    t_sampler = time_start()
    sampler = Sampler(sampler_factor)
    seq_after_sample = sampler.pre_process(seq_after_trim)
    # assert seq_after_sample.shape[1] == ceil(trimmer_seq_len / sampler_factor), \
    #     f"seq_after_sample.shape[1]={seq_after_sample.shape[1]}, " \
    #     f"ceil(trimmer_seq_len / sampler_factor)={ceil(trimmer_seq_len / sampler_factor)}"
    # assert seq_after_sample.shape == seq_after_trim.shape, \
    #     f"seq_after_sample.shape={seq_after_sample.shape}, seq_after_trim.shape={seq_after_trim.shape}"
    assert seq_after_sample.shape[1] == ceil(trimmer_seq_len / sampler_factor) or \
           seq_after_sample.shape == seq_after_trim.shape, \
        f"seq_after_sample.shape={seq_after_sample.shape}, seq_after_trim.shape={seq_after_trim.shape}, " \
        f"ceil(trimmer_seq_len / sampler_factor)={ceil(trimmer_seq_len / sampler_factor)}"
    del seq_after_trim
    log_time_delta(t_sampler, "sampler")

    t_inputer = time_start()
    inputer = Inputer(inputer_detect_method, inputer_fill_method, history_seqs)
    seq_after_input = inputer.pre_process(seq_after_sample)
    assert seq_after_input.shape == seq_after_sample.shape, \
        f"seq_after_input.shape={seq_after_input.shape}, seq_after_sample.shape={seq_after_sample.shape}"
    del seq_after_sample
    log_time_delta(t_inputer, "inputer")

    t_denoiser = time_start()
    denoiser = Denoiser(denoiser_method)
    seq_after_denoise = denoiser.pre_process(seq_after_input)
    assert seq_after_denoise.shape == seq_after_input.shape, \
        f"seq_after_denoise.shape={seq_after_denoise.shape}, seq_after_input.shape={seq_after_input.shape}"
    del seq_after_input
    log_time_delta(t_denoiser, "denoiser")

    # 选择数据的warping方法
    t_warper = time_start()
    warper = Warper(warper_method, clip_factor)
    seq_after_warp = warper.pre_process(seq_after_denoise)
    assert seq_after_warp.shape == seq_after_denoise.shape, \
        f"seq_after_warp.shape={seq_after_warp.shape}, seq_after_denoise.shape={seq_after_denoise.shape}"
    del seq_after_denoise
    log_time_delta(t_warper, "warper")

    t_decomposer = time_start()
    decomposer = Decomposer(decomposer_period, decomposer_components)
    seq_after_decompose = decomposer.pre_process(seq_after_warp)
    assert seq_after_decompose.shape == seq_after_warp.shape, \
        f"seq_after_decompose.shape={seq_after_decompose.shape}, seq_after_warp.shape={seq_after_warp.shape}"
    log_time_delta(t_decomposer, "decomposer")
    del seq_after_warp

    t_differentiator = time_start()
    differentiator = Differentiator(differentiator_n, clip_factor)
    seq_after_diff = differentiator.pre_process(seq_after_decompose)
    assert seq_after_diff.shape == seq_after_decompose.shape, \
        f"seq_after_diff.shape={seq_after_diff.shape}, seq_after_decompose.shape={seq_after_decompose.shape}"
    del seq_after_decompose
    log_time_delta(t_differentiator, "differentiator")

    t_normalizer = time_start()
    mode_for_scaler = 'train' if mode != 'test' else 'test'
    mode_scaler = dataset.get_mode_scaler(mode_for_scaler, normalizer_method, target_column)
    normalizer = Normalizer(normalizer_method, normalizer_mode, seq_after_diff, history_seqs, mode_scaler,
                            normalizer_ratio, clip_factor)
    seq_after_norm = normalizer.pre_process(seq_after_diff)
    assert seq_after_norm.shape == seq_after_diff.shape, \
        f"seq_after_norm.shape={seq_after_norm.shape}, seq_after_diff.shape={seq_after_diff.shape}"
    del seq_after_diff
    log_time_delta(t_normalizer, "normalizer")

    t_aligner = time_start()
    model_patch_len = model.patch_len
    aligner = Aligner(aligner_mode, aligner_method, data_patch_len=patch_len, model_patch_len=model_patch_len)
    seq_after_align = aligner.pre_process(seq_after_norm)
    assert seq_after_align.shape[1] % model_patch_len == 0 or seq_after_align.shape[1] % patch_len == 0 or \
           aligner_method == 'none' or aligner_mode == 'none', \
        f"seq_after_align.shape[1]={seq_after_align.shape[1]}, model_patch_len={model_patch_len}, patch_len={patch_len}"
    del seq_after_norm
    log_time_delta(t_aligner, "aligner")

    logging.info(f"seq_after_preprocess.shape={seq_after_align.shape}")
    seq_after_preprocess = seq_after_align

    # 将数据送进模型
    t_model = time_start()
    pred_len_needed = ceil(pred_len / sampler_factor / patch_len) * patch_len  # FIXME:小心

    tr_process.time_end()
    tr_model.time_start()

    try:
        preds = model.forcast(seq_after_preprocess, pred_len_needed)
        if np.isinf(preds).any() or np.isnan(preds).any():
            logging.error(f"preds has inf or nan!")
            my_clip(seq_after_preprocess, preds, nan_inf_clip_factor=nan_inf_clip_factor)
    except Exception as e:
        raise Exception(f"model.forcast(seq_after_preprocess, pred_len_needed) failed: {e}")

    tr_model.time_end()
    tr_process.time_start()

    assert preds.shape[1] == pred_len_needed, \
        f"preds.shape[1]={preds.shape[1]}, pred_len_needed={pred_len_needed}"
    del seq_after_preprocess
    log_time_delta(t_model, "model")

    preds = aligner.post_process(preds)

    t_normalizer = time_start()
    preds = normalizer.post_process(preds)
    log_time_delta(t_normalizer, "normalizer_back")

    t_differentiator = time_start()
    preds = differentiator.post_process(preds)
    log_time_delta(t_differentiator, "differentiator_back")

    t_decomposer = time_start()
    preds = decomposer.post_process(preds)
    log_time_delta(t_decomposer, "decomposer_back")

    t_warper = time_start()
    preds = warper.post_process(preds)
    log_time_delta(t_warper, "warper_back")

    preds = denoiser.post_process(preds)
    preds = inputer.post_process(preds)

    t_sampler = time_start()
    preds = sampler.post_process(preds)
    log_time_delta(t_sampler, "sampler_back")

    preds = trimmer.post_process(preds)

    tr_process.time_end()

    assert preds.shape[1] == pred_len, f"preds.shape[1]={preds.shape[1]}, pred_len={pred_len}"
    return preds, tr_process.get_total_duration(), tr_model.get_total_duration()
