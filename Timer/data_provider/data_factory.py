import os

from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from ..data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, \
    Dataset_Custom, Dataset_PEMS, UCRAnomalyloader, PSMSegLoader, MSLSegLoader, \
    SMAPSegLoader, SMDSegLoader, SWATSegLoader
from ..data_provider.data_loader_benchmark import CIDatasetBenchmark, \
    CIAutoRegressionDatasetBenchmark

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'PEMS': Dataset_PEMS,
    'custom': Dataset_Custom,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UCRA': UCRAnomalyloader,
}


def data_provider(args, flag):
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    # if args.task_name == 'forecast':
    if 'forecast' in args.task_name:
        if args.use_ims:
            data_set = CIAutoRegressionDatasetBenchmark(
                root_path=os.path.join(args.root_path, args.data_path),
                flag=flag,
                input_len=args.seq_len,
                label_len=args.label_len,
                pred_len=args.output_len if flag == 'test' else args.pred_len,
                data_type=args.data,
                scale=True,
                timeenc=timeenc,
                freq=args.freq,
                stride=args.stride,
                subset_rand_ratio=args.subset_rand_ratio,
            )
        else:
            data_set = CIDatasetBenchmark(
                root_path=os.path.join(args.root_path, args.data_path),
                flag=flag,
                input_len=args.seq_len,
                pred_len=args.pred_len,
                # pred_len=args.output_len if flag == 'test' else args.pred_len,
                data_type=args.data,
                scale=True,
                timeenc=timeenc,
                freq=args.freq,
                stride=args.stride,
                subset_rand_ratio=args.subset_rand_ratio,
                features=args.features,
            )
        if args.subset is not None and flag == 'train':
            assert isinstance(args.subset, int)
            data_set = Subset(data_set, indices=range(args.subset))
            print(f"subset: using the first {args.subset} samples in the dataloader")
        else:
            print("Using all samples in the dataloader")
            
        print(flag, len(data_set))
        if args.use_multi_gpu:
            train_datasampler = DistributedSampler(data_set, shuffle=shuffle_flag)
            data_loader = DataLoader(data_set,
                                     batch_size=args.batch_size,
                                     sampler=train_datasampler,
                                     num_workers=args.num_workers,
                                     persistent_workers=True,
                                     pin_memory=True,
                                     drop_last=False,
                                     )
        else:
            data_loader = DataLoader(
                data_set,
                batch_size=args.batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=False)
        return data_set, data_loader

    # elif args.task_name == 'anomaly_detection' or args.task_name == 'anomaly_detection_AEAR':
    elif 'anomaly_detection' in args.task_name:
        drop_last = False
        if args.data == 'UCRA':
            data_set = UCRAnomalyloader(
                args=args,
                root_path=args.root_path,
                data_path=args.data_path,
                seq_len=args.seq_len,
                patch_len=args.patch_len,
                flag=flag,
            )
        else:
            Data = data_dict[args.data]
            data_set = Data(
                args = args,
                root_path=args.root_path,
                win_size=args.seq_len,
                flag=flag,
            )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'imputation':
        Data = data_dict[args.data]
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    else:
        raise NotImplementedError
