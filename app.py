import pickle

import streamlit as st

from url_feature_extractor import extract_url_features


st.set_page_config(
    page_title="Phishing URL Detection",
    page_icon="🔐",
    layout="centered"
)


@st.cache_resource
def load_model_files():
    """
    Load the trained Logistic Regression model,
    feature order and phishing classification threshold.
    """
    with open("url_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("url_features.pkl", "rb") as feature_file:
        feature_order = pickle.load(feature_file)

    with open("url_threshold.pkl", "rb") as threshold_file:
        threshold = pickle.load(threshold_file)

    return model, feature_order, threshold


try:
    model, feature_order, phishing_threshold = load_model_files()

except FileNotFoundError:
    st.error(
        "One or more deployment files could not be found. "
        "Ensure url_model.pkl, url_features.pkl and "
        "url_threshold.pkl are available in the repository."
    )
    st.stop()

except Exception as error:
    st.error(
        f"The model files could not be loaded: {error}"
    )
    st.stop()


st.title("🔐 Phishing URL Detection System")

st.write(
    """
    Enter a complete website URL below. The application
    automatically extracts lexical and structural URL features
    and uses a trained Logistic Regression model to classify
    the URL as phishing or legitimate.
    """
)

st.info(
    "This system analyses the URL text only. It does not visit, "
    "download or inspect the submitted website."
)

url_input = st.text_input(
    "Website URL",
    placeholder="https://www.example.com"
)

predict_button = st.button(
    "Analyse URL",
    type="primary",
    use_container_width=True
)

if predict_button:

    if not url_input.strip():
        st.warning("Please enter a website URL.")

    else:
        try:
            input_features = extract_url_features(
                url_input,
                feature_order
            )

            probabilities = model.predict_proba(
                input_features
            )[0]

            probability_by_class = dict(
                zip(
                    model.classes_,
                    probabilities
                )
            )

            phishing_probability = float(
                probability_by_class.get(0, 0)
            )

            legitimate_probability = float(
                probability_by_class.get(1, 0)
            )

            # Apply the saved phishing threshold, currently 0.30
            prediction = (
                0
                if phishing_probability >= phishing_threshold
                else 1
            )

            st.subheader("Prediction Result")

            if prediction == 0:
                st.error(
                    "⚠️ Prediction: Phishing URL"
                )
            else:
                st.success(
                    "✅ Prediction: Legitimate URL"
                )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Phishing probability",
                    f"{phishing_probability:.2%}"
                )

            with col2:
                st.metric(
                    "Legitimate probability",
                    f"{legitimate_probability:.2%}"
                )

            with st.expander(
                "View extracted URL features"
            ):
                st.dataframe(
                    input_features.T.rename(
                        columns={0: "Value"}
                    ),
                    use_container_width=True
                )

            st.caption(
                "The prediction is produced by a machine-learning "
                "model and should not be treated as a guarantee. "
                "New or unusual URLs may be misclassified."
            )

        except ValueError as error:
            st.warning(str(error))

        except Exception as error:
            st.error(
                "An unexpected prediction error occurred: "
                f"{error}"
            )


st.divider()

st.markdown(
    f"""
    **Model:** Logistic Regression  
    **Input:** 14 lexical and structural URL features  
    **Phishing threshold:** {phishing_threshold:.2f}  
    **Output:** Phishing or legitimate prediction
    """
)
