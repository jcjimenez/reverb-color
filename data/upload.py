from glob import iglob
from os.path import join

from azure.storage.blob import BlockBlobService


def upload_to_blob(account_name, account_key, container_name, folder):
    blob = BlockBlobService(account_name, account_key)
    blob.create_container(container_name, public_access='container')
    for path in iglob(join(folder, '**', '*.*'), recursive=True):
        blob.create_blob_from_path(container_name, path, path)


def _main():
    from argparse import ArgumentParser

    parser = ArgumentParser(__doc__)
    parser.add_argument('storage_account_name')
    parser.add_argument('storage_account_key')
    parser.add_argument('container_name')
    parser.add_argument('folder')
    args = parser.parse_args()

    upload_to_blob(args.storage_account_name, args.storage_account_key,
                   args.container_name, args.folder)


if __name__ == '__main__':
    _main()
