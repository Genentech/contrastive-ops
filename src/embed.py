from pathlib import Path
from src.helper import get_module, embed_images
import argparse
import wandb
from lightning.pytorch.loggers import WandbLogger

def embed(run_id, run_name, version='best', loader_param=None, module='contrastive'):
    '''
    wandb_logger: the wandb logger object
    loader_param: the parameters for the dataloader
    '''
    # save run id and run name to file
    checkpoint_reference = f'wang-jerry/ops-training/model-{run_id}:{version}'
    artifact_dir = WandbLogger.download_artifact(artifact=checkpoint_reference)
    print(checkpoint_reference)

    # load checkpoint
    modelname = run_name.split('_')[0]
    ModelClass = get_module(modelname, 'model')
    DataClass = get_module(module, 'dataloader')

    checkpt_path = Path(artifact_dir) / "model.ckpt"
    model = ModelClass.load_from_checkpoint(checkpt_path)
    dm = DataClass.load_from_checkpoint(checkpt_path)

    embedding_df = embed_images(model, dm, stage='embed', loader_param=loader_param, modelname=module)
    embedding_df.to_pickle(f'/home/wangz222/scratch/embedding/{run_name}_{run_id}_{version}.pkl')

    wandb.finish() # end wandb run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding script with run_id and run_name arguments.")
    parser.add_argument("--run_id", type=str, help="The run ID.")
    parser.add_argument("--run_name", type=str, help="The run name.")
    parser.add_argument("-v", "--version", type=str, default="best")
    parser.add_argument("--batch_size", type=int, default=4200, help="Batch size for the loader.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for the loader.")
    parser.add_argument("--module", type=str, default="contrastive", help="The module name.")
    args = parser.parse_args()

    loader_param = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    embed(run_id=args.run_id, run_name=args.run_name, version=args.version, loader_param=loader_param, module=args.module)
