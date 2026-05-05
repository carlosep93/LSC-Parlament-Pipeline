from dotenv import dotenv_values
from pymongo import MongoClient


class Database:
    EMPTY_MESSAGE_ERROR = (
        "Cannot insert into %s: Empty content. Please provide data to insert."
    )

    def __init__(self, pipeline: str) -> None:
        self.pipeline = pipeline
        self.config = dotenv_values(".env")
        self.db_name = self.config["DATABASE_NAME"]

    def insert(self, content: dict):
        pass

    def find(self, key: str, value: any) -> bool:
        pass


class MongoDatabase(Database):
    def __init__(self, pipeline: str):
        super().__init__(pipeline)
        connection_string = self.config["MONGODB_CONNECTION_STRING"]
        self.database = MongoClient(connection_string)

    def insert(self, content: dict):
        if len(content) == 0:
            raise ValueError(self.EMPTY_MESSAGE_ERROR % "MongoDB")

        mongo_database = self.database[self.db_name][self.pipeline]
        mongo_database.insert_one(content)

    def find(self, key: str, value: any) -> bool:
        mongo_database = self.database[self.db_name][self.pipeline]
        result = mongo_database.find_one({key: value})

        if result:
            return True

        return False

    def renameCollection(self, old_collection: str, new_collection: str):
        mongo_database = self.database[self.db_name][old_collection]
        mongo_database.rename(new_collection)
