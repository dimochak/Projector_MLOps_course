import io

import lakefs_client
from lakefs_client import models
from lakefs_client.client import LakeFSClient
import pickle

from lakefs_client.exceptions import NotFoundException
from lakefs_client.model.repository_creation import RepositoryCreation


class LakeFS:
    def __init__(self, username, password):
        self.configuration = lakefs_client.Configuration()
        self.configuration.username = username
        self.configuration.password = password
        self.configuration.host = 'http://localhost:8000'

        self.client = LakeFSClient(self.configuration)

    def get_data(self, repository_name, object_name):
        """
        fetch object from bucket based on type
        :param repository_name: Name of repository in LakeFS: str
        :param object_name: name of object : str
        :return : Boolean
        """
        try:
            branch_name = 'main'
            object_data = self.client.objects.get_object(
                repository=repository_name,
                ref=branch_name,
                path=object_name)
            data = pickle.load(object_data)
            print("Object is loaded successfully.")
            return data
        except NotFoundException as error:
            print(f"Not able to get data from LakeFS: {error.body}")

    def insert_data(self, data, repository_name, object_name, create_new_repository=False):
        """
            insert object into repository
            :param data: data to upload
            :param repository_name: Repository name in LakeFS : str
            :param object_name: name of minio object : str
            :param create_new_repository: option to create new repository ("default value: False")
            :return: status : True or False
        """
        try:
            data = pickle.dumps(data)
            self.client.objects.upload_object(
                repository=repository_name,
                branch='main',
                path=object_name,
                content=io.BytesIO(data))
            print(f"Object has been successfully stored in {repository_name}")
            is_success = True
        except NotFoundException as error:
            print(f"Not able to insert data into LakeFS: {error.body}")
            if create_new_repository:
                self.client.repositories.create_repository(
                    RepositoryCreation(
                        name=repository_name, storage_namespace='main'
                    )
                )
                print("Repository created successfully for saving an object..")
                self.insert_data(data, repository_name, object_name)
                is_success = True
            else:
                is_success = False
        return is_success

    def delete_data(self, repository_name, object_name):
        """
        delete object from repository based on type
        :param bucket_name: Repository name in LakeFS : str
        :param object_name: name of LakeFS object : str
        :return: status : True or False
        """
        try:
            self.client.objects.delete_object(repository=repository_name,
                                              branch='main',
                                              path=object_name)
            print("Object deleted successfully.")
        except NotFoundException as error:
            print(f"Object can not be deleted: {error.body}")


if __name__ == '__main__':
    client = LakeFS(username='AKIAJ7VJQHGEEPTE2CJQ',
                    password='Kfmsi5H96Qtv961z8jKA+v9yrljQgwRqB9DlSzJB')
    repository_name = 'test-repository'
    file_name = 'list_example'
    list_example = [1, 2, 3]

    # Testing:
    # 1) Get data from empty bucket
    obj = client.get_data(repository_name=repository_name,
                          object_name=file_name)
    print(f'Get data from empty bucket: {obj}')

    # 2) Put new data and retrieve it back
    client.insert_data(data=list_example, repository_name=repository_name, object_name=file_name)
    obj = client.get_data(repository_name=repository_name,
                          object_name=file_name)
    print(f'Get data from bucket: {obj}')

    # 3) Update object: insert it once again with the same object_name
    list_example.append(4)
    client.insert_data(data=list_example, repository_name=repository_name, object_name=file_name)
    obj = client.get_data(repository_name=repository_name,
                          object_name=file_name)
    print(f'Updated data from bucket: {obj}')

    # 4) Delete object
    client.delete_data(repository_name=repository_name,
                       object_name=file_name)
    obj = client.get_data(repository_name=repository_name,
                          object_name=file_name)
    print(f'Get data from empty bucket: {obj}')