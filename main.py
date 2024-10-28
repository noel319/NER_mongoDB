import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from aiomultiprocess import Pool
import spacy

# Load SpaCy NLP model for both English and Russian
nlp_en = spacy.load("en_core_web_sm")
nlp_ru = spacy.load("ru_core_news_sm")

# MongoDB configuration
DATABASE_NAME = "migrate_database"
COLLECTION_COUNT = 14000
MONGO_URI = "mongodb://localhost:27017"

# Define chunk size for processing
CHUNK_SIZE = 10000  # Adjust as needed based on memory limits

# Connect to MongoDB
client = AsyncIOMotorClient(MONGO_URI)
db = client[DATABASE_NAME]

async def detect_full_name(doc):
    """Detects full names or partial names in the document fields."""
    name_fields = []
    
    # Check each field for name entities in both English and Russian
    for key, value in doc.items():
        if isinstance(value, str):  # Apply NLP only to string fields
            doc_en = nlp_en(value)
            doc_ru = nlp_ru(value)
            
            if any(ent.label_ == "PERSON" for ent in doc_en.ents) or any(ent.label_ == "PER" for ent in doc_ru.ents):
                name_fields.append((key, value))
                
    return name_fields

async def update_chunk(collection, chunk):
    """Update documents in a chunk based on detected name fields."""
    for doc in chunk:
        name_fields = await detect_full_name(doc)
        
        if not name_fields:
            continue  # Skip if no name fields detected

        # Define the update query and filter based on the detection
        if len(name_fields) == 1:
            # Single field detected, rename it to 'full_name'
            full_name_field = name_fields[0][0]
            update_query = {"$rename": {full_name_field: "full_name"}}
            update_filter = {full_name_field: {"$exists": True}, "full_name": {"$exists": False}}
        else:
            # Multiple name fields detected, combine into 'full_name'
            full_name = " ".join([name for _, name in name_fields])
            update_query = {
                "$set": {"full_name": full_name},
                "$unset": {field: "" for field, _ in name_fields}
            }
            update_filter = {name_fields[0][0]: {"$exists": True}, "full_name": {"$exists": False}}
        
        # Update documents matching the filter in bulk for this chunk
        result = await collection.update_many(update_filter, update_query)
        print(f"Updated {result.modified_count} documents in this chunk of collection.")

async def process_collection(collection_name):
    """Processes a single collection in chunks."""
    collection = db[collection_name]
    total_docs = await collection.count_documents({})  # Total documents in the collection
    processed_docs = 0

    while processed_docs < total_docs:
        # Fetch a chunk of documents to process
        chunk = await collection.find().skip(processed_docs).limit(CHUNK_SIZE).to_list(length=CHUNK_SIZE)

        if not chunk:
            break  # Exit if no more documents to process

        # Process and update documents in the current chunk
        await update_chunk(collection, chunk)
        processed_docs += len(chunk)
        print(f"Processed {processed_docs} / {total_docs} documents in collection '{collection_name}'.")

async def main():
    """Main function to coordinate collection processing."""
    collection_names = await db.list_collection_names()

    async with Pool() as pool:
        await pool.map(process_collection, collection_names[:COLLECTION_COUNT])

if __name__ == "__main__":
    asyncio.run(main())
