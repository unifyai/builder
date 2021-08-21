# local
from ivy_builder.specs.data_loader_spec import DataLoaderSpec


class JSONDataLoaderSpec(DataLoaderSpec):

    def __init__(self, dataset_spec, batch_size, window_size, starting_idx, num_sequences,
                 num_workers=1, cache_size=0, unused_key_chains=None, preload_containers=False, preshuffle_data=True,
                 shuffle_buffer_size=0, with_prefetching=True, post_proc_fn=None, prefetch_to_gpu=False,
                 single_pass=False, array_strs=None, float_strs=None, uint8_strs=None, custom_img_strs=None,
                 custom_img_fns=None, custom_strs=None, custom_fns=None, array_mode='pickled', **kwargs):

        unused_key_chains = [] if unused_key_chains is None else unused_key_chains
        array_strs = [] if array_strs is None else array_strs
        float_strs = [] if float_strs is None else float_strs
        uint8_strs = [] if uint8_strs is None else uint8_strs
        custom_img_strs = [[]] if custom_img_strs is None else custom_img_strs
        custom_img_fns = [] if custom_img_fns is None else custom_img_fns
        custom_strs = [[]] if custom_strs is None else custom_strs
        custom_fns = [] if custom_fns is None else custom_fns

        super(JSONDataLoaderSpec, self).__init__(dataset_spec,
                                                 batch_size=batch_size,
                                                 window_size=window_size,
                                                 starting_idx=starting_idx,
                                                 num_sequences=num_sequences,
                                                 num_workers=num_workers,
                                                 cache_size=cache_size,
                                                 unused_key_chains=unused_key_chains,
                                                 preload_containers=preload_containers,
                                                 preshuffle_data=preshuffle_data,
                                                 shuffle_buffer_size=shuffle_buffer_size,
                                                 with_prefetching=with_prefetching,
                                                 post_proc_fn=post_proc_fn,
                                                 prefetch_to_gpu=prefetch_to_gpu,
                                                 single_pass=single_pass,
                                                 array_strs=array_strs,
                                                 float_strs=float_strs,
                                                 uint8_strs=uint8_strs,
                                                 custom_img_strs=custom_img_strs,
                                                 custom_img_fns=custom_img_fns,
                                                 custom_strs=custom_strs,
                                                 custom_fns=custom_fns,
                                                 array_mode=array_mode,
                                                 **kwargs)
