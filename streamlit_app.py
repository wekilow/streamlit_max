import streamlit as st

st.title("OpenCV Streamlit Demo")
st.header("header of Streamlit Demo")

image = st.file_uploader("Upload an image file")
# st.image(image)

st.text("this is text area")
selected_value = st.selectbox("Select Box", ["None", "Filter1", "Filter2"])
st.write(selected_value)

checkbox_value = st.checkbox("Checkbox")
st.write(checkbox_value)