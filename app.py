import pickle
import re
from urllib.parse import urlparse

import streamlit as st

from url_feature_extractor import extract_url_features


st.set_page_config(
    page_title="Phishing URL Detection",
    page_icon="🔐",
    layout="centered"
)


@st.cache_resource
def load_deployment_files():
    """
    Load the model, feature order, classification threshold,
    and popular-domain reference list.
    """
    with open("url_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("url_features.pkl", "rb") as feature_file:
        feature_order = pickle.load(feature_file)

    with open("url_threshold.pkl", "rb") as threshold_file:
        threshold = pickle.load(threshold_file)

    with open("top_domains.pkl", "rb") as domains_file:
        popular_domains = set(pickle.load(domains_file))

    return model, feature_order, threshold, popular_domains


def extract_hostname(url):
    """
    Extract and normalise the hostname for the popularity check.
    """
    cleaned_url = url.strip()

    if not re.match(
        r"^[a-zA-Z][a-zA-Z0-9+.-]*://",
        cleaned_url
    ):
        cleaned_url = "https://" + cleaned_url

    parsed_url = urlparse(cleaned_url)

    hostname = (
        parsed_url.hostname or ""
    ).lower().strip(".")

    if hostname.startswith("www."):
        hostname = hostname[4:]

    return hostname


try:
    (
        model,
        feature_order,
        phishing_threshold,
        popular_domains
    ) = load_deployment_files()

except FileNotFoundError:
    st.error(
        "A required deployment file is missing. Ensure that "
        "url_model.pkl, url_features.pkl, url_threshold.pkl, "
        "and top_domains.pkl are available in the repository."
    )
    st.stop()

except Exception as error:
    st.error(
        f"The deployment files could not be loaded: {error}"
    )
    st.stop()


st.title("🔐 Phishing URL Detection System")

st.write(
    """
    Enter a complete website URL below. The system analyses
    lexical and structural characteristics of the URL using a
    Logistic Regression model.
    """
)

st.info(
    "The application analyses URL text only. It does not open, "
    "download, or inspect the submitted website."
)

url_input = st.text_input(
    "Website URL",
    placeholder="https://www.example.com"
)

analyse_button = st.button(
    "Analyse URL",
    type="primary",
    use_container_width=True
)


if analyse_button:

    if not url_input.strip():
        st.warning("Please enter a website URL.")

    else:
        try:
            input_features = extract_url_features(
                url_input,
                feature_order
            )

            hostname = extract_hostname(url_input)

            if not hostname:
                raise ValueError(
                    "The submitted URL does not contain "
                    "a valid hostname."
                )

            probabilities = model.predict_proba(
                input_features
            )[0]

            probability_by_class = dict(
                zip(model.classes_, probabilities)
            )

            phishing_score = float(
                probability_by_class.get(0, 0)
            )

            legitimate_score = float(
                probability_by_class.get(1, 0)
            )

            model_flags_phishing = (
                phishing_score >= phishing_threshold
            )

            popular_domain_signal = (
                hostname in popular_domains
            )

            st.subheader("Analysis Result")

            if model_flags_phishing and popular_domain_signal:
                st.warning(
                    "⚠️ Conflicting signals — manual review required"
                )

                st.write(
                    "The hostname appears in the popular-domain "
                    "reference list, but the URL structure received "
                    "a high phishing score. This may happen with "
                    "long search, tracking, or authentication URLs."
                )

            elif model_flags_phishing:
                st.error(
                    "🚨 Result: Likely phishing"
                )

            else:
                st.success(
                    "✅ Result: Likely legitimate"
                )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Model phishing score",
                    f"{phishing_score:.2%}"
                )

            with col2:
                st.metric(
                    "Model legitimate score",
                    f"{legitimate_score:.2%}"
                )

            st.write(
                f"**Hostname analysed:** `{hostname}`"
            )

            st.write(
                "**Popular-domain signal:** "
                + (
                    "Recognised"
                    if popular_domain_signal
                    else "Not recognised"
                )
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
                "The displayed scores are model outputs, not proof "
                "that a website is safe or malicious. Popular-domain "
                "membership is an additional signal only and does "
                "not guarantee that every page is trustworthy."
            )

        except ValueError as error:
            st.warning(str(error))

        except Exception as error:
            st.error(
                "An unexpected analysis error occurred: "
                f"{error}"
            )


st.divider()

st.markdown(
    f"""
    **Model:** Logistic Regression  
    **Input:** 14 lexical and structural URL features  
    **Phishing threshold:** {phishing_threshold:.2f}  
    **Safety layer:** Popular-domain conflict detection  
    **Possible outputs:** Likely legitimate, likely phishing, or manual review
    """
)
