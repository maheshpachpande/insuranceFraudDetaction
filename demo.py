from pymongo import MongoClient
import pandas as pd





client = MongoClient.("mongodb+srv://pachpandemahesh300:n5wQ5jQerIDjAucn@cluster0.ihqknkm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

db = client["insurance"]
collection = db["data"]

df = pd.read_csv("data.csv")
insert_many_result = collection.insert_many(df.to_dict("records"))

print("Data inserted successfully into MongoDB collection.")
print("Number of records inserted:", len(df))
print("First few records:")
print(df.head())
print("MongoDB connection established successfully.")