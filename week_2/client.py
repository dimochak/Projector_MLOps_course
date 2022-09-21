from minio import Minio
from minio.error import S3Error
import io
import pickle


class MinioDB:
    def __init__(self, host, access_key, secret_key):
        # Initialize minioClient
        self.host = host
        self.access_key = access_key
        self.secret_key = secret_key
        try:
            self.client = Minio(self.host,
                                access_key=self.access_key, secret_key=self.secret_key,
                                secure=False)
        except S3Error as error:
            print(error)
            raise

    def get_data(self, bucket_name, object_name):
        """
        fetch object from bucket based on type
        :param bucket_name: Container name in Minio : str
        :param object_name: name of minio object : str
        :return: dataframe : Boolean
        """
        try:
            """checking whether the bucket exists or not"""
            bucket = self.client.bucket_exists(bucket_name)
            if bucket:
                object_data = self.client.get_object(bucket_name, object_name)
                data = pickle.load(object_data)
                print("Object is loaded successfully.")
                return data
            else:
                print("Bucket does not exist")
        except S3Error as error:
            print(f"Not able to get data from minio: {error}")

    def insert_data(self, data, bucket_name, object_name, create_new_bucket=False):
        """
                insert object into bucket based on type
                :param bucket_name: Container name in Minio : str
                :param object_name: name of minio object : str
                :param create_new_bucket: option to create new bucket ("default value: False")
                :return: status : True or False
                """
        try:
            bucket = self.client.bucket_exists(bucket_name)
            is_success = False
            if bucket:
                data = pickle.dumps(data)
                self.client.put_object(bucket_name, object_name, data=io.BytesIO(data), length=len(data))
                print(f"Object has been successfully stored in {bucket_name}")
                is_success = True
            elif create_new_bucket:
                self.client.make_bucket(bucket_name)
                print("Bucket created successfully for saving a model..")
                self.insert_data(data, bucket_name, object_name)
            return is_success
        except S3Error as error:
            print(f"Not able to insert data into minio: {error}")

    def delete_data(self, bucket_name, object_name):
        """
        delete object from bucket based on type
        :param bucket_name: Container name in Minio : str
        :param object_name: name of minio object : str
        :return: status : True or False
        """
        try:
            bucket = self.client.bucket_exists(bucket_name)
            is_success = False
            if bucket:
                # deleting/removing the previously trained model from the minio database based on the selection...
                self.client.remove_object(bucket_name, object_name)
                print("Object deleted successfully.")
                is_success = True

            else:
                print("Object can't be removed because bucket is not available.")
            return is_success

        except S3Error as error:
            print(f"Object can not be deleted: {error}")


if __name__ == '__main__':
    client = MinioDB('127.0.0.1:9000', access_key='dpekach', secret_key='bestadmin')
    bucket_name = 'test-bucket'
    file_name = 'list_example'
    list_example = [1, 2, 3]

    # Testing:
    # 1) Get data from empty bucket
    obj = client.get_data(bucket_name="test-bucket",
                          object_name=file_name)
    print(f'Get data from empty bucket: {obj}')

    # 2) Put new data and retrieve it back
    client.insert_data(data=list_example, bucket_name='test-bucket', object_name=file_name)
    obj = client.get_data(bucket_name="test-bucket",
                          object_name=file_name)
    print(f'Get data from bucket: {obj}')

    # 3) Update object: insert it once again with the same object_name
    list_example.append(4)
    client.insert_data(data=list_example, bucket_name='test-bucket', object_name=file_name)
    obj = client.get_data(bucket_name="test-bucket",
                          object_name=file_name)
    print(f'Updated data from bucket: {obj}')

    # 4) Delete object
    client.delete_data(bucket_name='test-bucket', object_name=file_name)
    obj = client.get_data(bucket_name="test-bucket",
                          object_name=file_name)
    print(f'Get data from empty bucket: {obj}')

