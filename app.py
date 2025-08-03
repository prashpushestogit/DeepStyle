import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
import faiss
import boto3

# Load AWS secrets (ensure your .streamlit/secrets.toml is set up)
aws_access_key = st.secrets["aws"]["AWS_ACCESS_KEY_ID"]
aws_secret_key = st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"]
region = st.secrets["aws"]["AWS_REGION"]
bucket = st.secrets["aws"]["S3_BUCKET"]

# Connect to S3
s3 = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=region
)

# Load features and filenames
def load_pickle_from_s3(bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    return pickle.load(response['Body'])

feature_list = np.array(load_pickle_from_s3(bucket, "features.pkl"))
filenames = load_pickle_from_s3(bucket, "filesname.pkl")

# Load the ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        save_dir = 'uploads'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, uploaded_file.name)
        with open(save_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        # Optional: upload to S3 as well
        s3.upload_file(save_path, bucket, f"user_uploads/{uploaded_file.name}")
        return save_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img).astype('float32')
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list, k=6):
    features = np.array(features).astype('float32')
    feature_list = np.array(feature_list).astype('float32')
    index = faiss.IndexFlatL2(feature_list.shape[1])
    index.add(feature_list)
    distances, indices = index.search(features.reshape(1, -1), k)
    return indices

# MAIN LOGIC
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    save_path = save_uploaded_file(uploaded_file)
    if save_path:
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption='Uploaded Image')
        features = feature_extraction(save_path, model)
        indices = recommend(features, feature_list)
        st.subheader("Top 5 Recommendations")
        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                # Convert backslashes to forward slashes for S3 compatibility
                s3_key = filenames[indices[0][i]].replace("\\", "/")
                url = s3.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": bucket, "Key": s3_key},
                    ExpiresIn=3600
                )
                st.image(url)
    else:
        st.error("Some error occurred during file upload.")
