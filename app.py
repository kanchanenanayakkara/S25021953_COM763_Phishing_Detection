import ipaddress
import pickle
import re
from urllib.parse import urlparse

import streamlit as st
import tldextract

from url_feature_extractor import extract_url_features


st.set_page_config(
    page_title="Phishing URL Risk Assessment",
    page_icon="🔐",
    layout="centered"
)


# Use the bundled Public Suffix List snapshot.
# This avoids downloading suffix data while the app is running.
domain_extractor = tldextract.TLDExtract(
    suffix_list_urls=()
)


@st.cache_resource
def load_deployment_files():
    """
    Load the model, feature order, threshold and
    popular-domain reference list.
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


def normalise_for_parsing(url):
    """
    Add a scheme when one is missing so that urllib can
    correctly identify the hostname.
    """
    cleaned_url = url.strip()

    if not re.match(
        r"^[a-zA-Z][a-zA-Z0-9+.-]*://",
        cleaned_url
    ):
        cleaned_url = "https://" + cleaned_url

    return cleaned_url


def extract_domain_information(url):
    """
    Extract the hostname, registered domain and subdomain.

    Example:
    google.com.evil-site.com
        hostname: google.com.evil-site.com
        registered domain: evil-site.com
        subdomain: google.com
    """
    normalised_url = normalise_for_parsing(url)
    parsed_url = urlparse(normalised_url)

    hostname = (
        parsed_url.hostname or ""
    ).lower().strip(".")

    if not hostname:
        raise ValueError(
            "The submitted URL does not contain a valid hostname."
        )

    extracted = domain_extractor(hostname)

    registered_domain = ".".join(
        part
        for part in [
            extracted.domain,
            extracted.suffix
        ]
        if part
    )

    subdomain = extracted.subdomain or ""

    # IP addresses do not have a normal registered-domain structure.
    if is_ip_address(hostname):
        registered_domain = hostname
        subdomain = ""

    return {
        "hostname": hostname,
        "registered_domain": registered_domain,
        "subdomain": subdomain,
        "normalised_url": normalised_url
    }


def is_ip_address(hostname):
    """
    Return True when the hostname is an IPv4 or IPv6 address.
    """
    try:
        ipaddress.ip_address(
            hostname.strip("[]")
        )
        return True
    except ValueError:
        return False


def find_popular_domain_in_subdomain(
    subdomain,
    popular_domains
):
    """
    Detect whether a popular registered domain has been placed
    deceptively inside the subdomain.

    Example:
    google.com.evil-site.com
    """
    if not subdomain:
        return None

    labels = subdomain.lower().split(".")

    for index in range(len(labels)):
        candidate = ".".join(labels[index:])

        if candidate in popular_domains:
            return candidate

    return None


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
        "url_model.pkl, url_features.pkl, url_threshold.pkl "
        "and top_domains.pkl are available."
    )
    st.stop()

except Exception as error:
    st.error(
        f"The deployment files could not be loaded: {error}"
    )
    st.stop()


st.title("🔐 Phishing URL Risk Assessment System")

st.write(
    """
    Enter a complete website URL below. The system combines a
    Logistic Regression model with domain-structure and
    popular-domain safety checks.
    """
)

st.info(
    "The application analyses URL text only. It does not open, "
    "download or inspect the submitted website."
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

            domain_info = extract_domain_information(
                url_input
            )

            hostname = domain_info["hostname"]
            registered_domain = domain_info[
                "registered_domain"
            ]
            subdomain = domain_info["subdomain"]
            normalised_url = domain_info[
                "normalised_url"
            ]

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

            registered_domain_is_popular = (
                registered_domain in popular_domains
            )

            imitated_popular_domain = (
                find_popular_domain_in_subdomain(
                    subdomain,
                    popular_domains
                )
            )

            ip_address_signal = is_ip_address(
                hostname
            )

            at_symbol_signal = (
                "@" in normalised_url
            )

            punycode_signal = (
                "xn--" in hostname
            )

            strong_deception_signal = any([
                ip_address_signal,
                at_symbol_signal,
                imitated_popular_domain is not None
            ])

            st.subheader("Analysis Result")

            if strong_deception_signal:
                st.error(
                    "🚨 Result: Likely phishing"
                )

                reasons = []

                if ip_address_signal:
                    reasons.append(
                        "The hostname is an IP address."
                    )

                if at_symbol_signal:
                    reasons.append(
                        "The URL contains an @ symbol, which can "
                        "hide the real destination."
                    )

                if imitated_popular_domain:
                    reasons.append(
                        f"The subdomain imitates the recognised "
                        f"domain `{imitated_popular_domain}`, while "
                        f"the actual registered domain is "
                        f"`{registered_domain}`."
                    )

                for reason in reasons:
                    st.write(f"- {reason}")

            elif (
                model_flags_phishing
                and registered_domain_is_popular
            ):
                st.warning(
                    "⚠️ Conflicting signals — manual review required"
                )

                st.write(
                    "The registered domain appears in the "
                    "popular-domain reference list, but the URL "
                    "structure received a high phishing score. "
                    "This can occur with long search, tracking or "
                    "authentication URLs."
                )

            elif model_flags_phishing:
                st.error(
                    "🚨 Result: Likely phishing"
                )

            elif registered_domain_is_popular:
                st.success(
                    "✅ Result: Likely legitimate"
                )

            else:
                st.warning(
                    "⚠️ Unverified domain — manual review recommended"
                )

                st.write(
                    "The model produced a low phishing score, but "
                    "the registered domain was not found in the "
                    "popular-domain reference list."
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
                f"**Hostname:** `{hostname}`"
            )

            st.write(
                f"**Registered domain:** "
                f"`{registered_domain}`"
            )

            if subdomain:
                st.write(
                    f"**Subdomain:** `{subdomain}`"
                )

            st.write(
                "**Popular registered-domain signal:** "
                + (
                    "Recognised"
                    if registered_domain_is_popular
                    else "Not recognised"
                )
            )

            if punycode_signal:
                st.warning(
                    "The hostname contains internationalised-domain "
                    "encoding (`xn--`). Review it carefully."
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
                "Scores are machine-learning outputs, not proof "
                "that a website is safe or malicious. Popular-domain "
                "membership and structural checks are supporting "
                "signals only."
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
    **Safety checks:** Registered-domain parsing, popular-domain
    conflict detection and deceptive-subdomain detection  
    **Possible outputs:** Likely legitimate, likely phishing,
    unverified domain or manual review
    """
)
