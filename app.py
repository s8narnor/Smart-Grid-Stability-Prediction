import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the saved model
loaded_model = joblib.load('sgs-model.sav')

# Function to get user input for a single data point
def get_input():
    input_data = []
    features = ['tau1', 'tau2', 'tau3', 'tau4', 'p1', 'p2', 'p3', 'p4', 'g1', 'g2', 'g3', 'g4']
    
    st.write("""##### Enter Input Parameters""")
    
    # Create 3 columns (since we need 4 inputs per row, we can use 3 columns per row with an input field in each)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        input_data.append(st.number_input(f"{features[0]}:", value=0.0, format="%.4f"))
        input_data.append(st.number_input(f"{features[1]}:", value=0.0, format="%.4f"))
        input_data.append(st.number_input(f"{features[4]}:", value=0.0, format="%.4f"))
        input_data.append(st.number_input(f"{features[5]}:", value=0.0, format="%.4f"))
    
    with col2:
        input_data.append(st.number_input(f"{features[2]}:", value=0.0, format="%.4f"))
        input_data.append(st.number_input(f"{features[3]}:", value=0.0, format="%.4f"))
        input_data.append(st.number_input(f"{features[6]}:", value=0.0, format="%.4f"))
        input_data.append(st.number_input(f"{features[7]}:", value=0.0, format="%.4f"))
    
    with col3:
        input_data.append(st.number_input(f"{features[8]}:", value=0.0, format="%.4f"))
        input_data.append(st.number_input(f"{features[9]}:", value=0.0, format="%.4f"))
        input_data.append(st.number_input(f"{features[10]}:", value=0.0, format="%.4f"))
        input_data.append(st.number_input(f"{features[11]}:", value=0.0, format="%.4f"))
    
    return np.array(input_data).reshape(1, -1)

# Streamlit app framework
st.set_page_config(page_title="Smart Grid Stability Prediction", page_icon="⚡", layout="centered")

# Create a sidebar menu with options (tabs)
menu = ["Home", "Prediction", "About"]
choice = st.sidebar.selectbox("Menu", menu)

# Home Tab
if choice == "Home":
    st.sidebar.image("images/robotic-hand.png",  use_container_width=True )

    st.title("Smart Grid Stability⚡")
    
    # Upload your smart grid-related image here
    st.image('images/smart-grids.jpg', caption='Smart Grid and Stability', use_container_width=True)

    st.subheader("Project Overview:")
    st.write("""
    #### Understanding Smart Grids:
    Grid stability refers to the ability of an electrical power grid to maintain a balance between energy production and consumption while responding effectively to fluctuations in voltage and frequency. Traditional power grids, often fueled by non-renewable energy sources like coal, oil, and natural gas, rely on centralized systems and synchronous generators to maintain this balance, making stability relatively straightforward to achieve.
    However, with the increasing integration of renewable energy sources, such as wind and solar, the dynamics of grid stability have become more complex. Unlike traditional energy systems, renewable energy is inherently variable and intermittent, requiring advanced mechanisms to ensure stability.

    For a grid to remain stable, energy supply must equal energy demand. Any imbalance—whether excess energy generation or higher energy consumption—can lead to frequency disturbances, voltage volatility, or even power outages. Addressing these challenges requires rapid adjustments to restore equilibrium, making the stability of modern grids essential for a reliable power supply.
    The future of energy grids lies in innovative solutions that can adapt to renewable energy sources, ensuring stability while driving sustainability.

    #### Renewable Energy Sources and Smart Grids:
    The rise of **renewable energy sources** offers a sustainable alternative to fossil fuels but brings new challenges:
    
    - **Prosumers**: Consumers who both produce and consume energy, making smart grids **bidirectional**.
    - **Complex Supply-Demand Management**: With more flexible generation sources, managing supply-demand and pricing decisions has become more challenging.
    
    These changes highlight the importance of **smart grid stability**.

    #### Objective:
    The goal of this project is to predict the stability of a smart grid based on various input parameters including reaction times, power distribution, and price elasticity coefficients for each participant.

    The stability of the grid depends on these values, and the prediction is made using a trained machine learning model.

    #### Key Features:
    - Predict whether a smart grid is stable or unstable based on user inputs.
    - Visualize key data insights such as correlations and distributions of power.
    - Simple user interface for ease of prediction.

    #### Data:
    The data used for training the model includes parameters like:

    - **`tau1`** to **`tau4`**: Reaction time of each network participant (in seconds), a real value within the range **0.5 to 10**.
        - `tau1`: Corresponds to the supplier node.
        - `tau2` to `tau4`: Correspond to the consumer nodes.

    - **`p1`** to **`p4`**: Nominal power produced (positive) or consumed (negative) by each network participant (in megawatts), a real value within the range:  
        - `p1` (supplier node): Positive values, indicating power supplied.  
        - `p2` to `p4` (consumer nodes): Negative values, within the range **-2.0 to -0.5**.  
        - **Power Balance**: The total power consumed is equal to the total power generated: **`p1 = - (p2 + p3 + p4)`**.

    - **`g1`** to **`g4`**: Price elasticity coefficients (gamma), a real value within the range **0.05 to 1.00**.  
        - `g1`: Corresponds to the supplier node.  
        - `g2` to `g4`: Correspond to the consumer nodes.
        
    #### Modeling Grid Stability:
    In **smart grids**, consumer demand data is evaluated against supply conditions, and pricing information is sent back to consumers. The objective is to dynamically estimate grid stability by considering both technical and economic factors. Researchers focus on the **Decentral Smart Grid Control (DSGC)** methodology, which monitors **grid frequency**, an important indicator of grid stability.
    
    - **Frequency**: AC frequency (in Hz) increases during excess generation and decreases during underproduction. This measurement provides critical information about power balance.
    - **DSGC Model**: Evaluates grid stability for a 4-node star architecture, accounting for:
      - **Power balance** (nominal power at each node)
      - **Reaction time** (how quickly participants adjust consumption/production in response to price)
      - **Energy price elasticity**
    """)
    
    # Add an image related to "Modeling Grid Stability"
    st.image('images/modeling-grid.png', caption='Modeling Grid Stability', use_container_width=True)

    st.write("""
    #### Addressing Simplifications in the Model:
    While the mathematical model for predicting grid instability exists, its execution relies on significant simplifications. The DSGC model, based on differential equations, is often simulated with fixed values for some variables and fixed distributions for others. This leads to two main issues: **"fixed inputs issue"** and **"equality issue"**.

    To overcome these simplifications, **machine learning** techniques, such as **decision trees (CART)** and **space-filling designs**, are used. Here's how it works:
    
    1. Input parameters (a vector) are fed into the DSGC model.
    2. The model outputs a binary result: **stable** or **unstable**.
    3. This process is repeated multiple times to generate a dataset of inputs and their respective outputs.
    4. The resulting dataset is used by machine learning models to make predictions about grid stability.
    
    As a result, **CART-based learning** achieves accuracy rates of **around 80%** in predicting grid stability or instability.
    """)

# Prediction Tab: Predict stability of smart grids
elif choice == "Prediction":
    st.sidebar.image("images/forecast.png",  use_container_width=True )
    st.title("Using Deep Learning to predict Smart Grid stability⚡")
    
    st.write("""
    #### Steps to Predict Smart Grid Stability 🧠:
    1. Enter the values for the required parameters (`tau1` to `tau4`, `p1` to `p4`, `g1` to `g4`) in the sidebar.
    2. Click on the **'Predict Stability'** button.
    3. View the prediction result, which will indicate whether the grid is **Stable** or **Unstable**.
    """)
    
    # Get user input
    user_input = get_input()

    # Check if all inputs are provided
    if np.any(user_input == 0):
        st.error("Invalid Input: Please try filling all values in the sidebar to proceed.")
    else:
        # Make prediction
        if st.button("Predict Stability"):
            with st.spinner("Predicting... Please wait!"):
                prediction = loaded_model.predict(user_input)

            # Display the result
            st.success("Prediction Complete!")
            if prediction[0] == 0:
                st.error("Prediction: **Unstable** ⚠️")
                st.write("""
                ### Recommendations for Unstable Grid:
                - **Improve Reaction Time (τ):** Ensure faster response to fluctuations by reducing the reaction times ('tau1' to 'tau4').
                - **Balance Power Distribution (P):** Reallocate power distribution among participants to reduce overloading or shortages.
                - **Adjust Price Elasticity (γ):** Optimize the elasticity coefficients ('g1' to 'g4') to improve the grid's economic stability.
                - **Monitor Grid Frequency:** Regularly monitor the AC frequency and implement corrective measures to maintain balance.
                """)
            else:
                st.success("Prediction: **Stable** ✔️")
                st.write("""
                ### Recommendations for Stable Grid:
                - **Maintain Current Parameters:** Ensure that reaction times, power distributions, and elasticity coefficients remain within optimal ranges.
                - **Enhance Monitoring Systems:** Continue monitoring grid performance to proactively address potential instability.
                - **Incorporate Renewable Energy:** Consider integrating additional renewable sources while maintaining balance.
                - **Invest in Smart Grid Technology:** Leverage advanced control systems to sustain stability as grid complexity grows.
                """)


# About Tab: Project objectives and details
elif choice == "About":
    st.sidebar.image("images/about-project.png",  use_container_width=True )
    st.sidebar.markdown(
    """
    <h3>About me</h3>
    <ul>
        <li><strong>📞 Contact Number:</strong> +91 9834958880</li>
        <li><strong>📧 Email:</strong> sarthaknarnor@gmail.com</li>
        <li><strong>🔗 LinkedIn:</strong> <a href="https://www.linkedin.com/in/sarthaknarnor/" target="_blank">Sarthak Narnor</a> 🔗</li>
    </ul>
    <br>
    <p><strong>Thank you for visiting!</strong></p>
    """, 
    unsafe_allow_html=True
    )

    st.subheader("About the Project 🚀")

    st.write("""
    
    This project uses the **"Electrical Grid Stability Simulated Dataset"** created by Vadim Arzamasov from the **Karlsruher Institut für Technologie** and donated to the **UCI Machine Learning Repository**. The dataset is publicly available [here](https://archive.ics.uci.edu/dataset/471/electrical+grid+stability+simulated+data).
    
    Two key references support this work:
    
    - **"Taming Instabilities in Power Grid Networks by Decentralized Control"** (B. Schäfer et al., 2016) explores the **Decentral Smart Grid Control (DSGC)** model for assessing smart grid stability.
    - **"Towards Concise Models of Grid Stability"** (V. Arzamasov et al., 2018) addresses data-mining techniques to simplify DSGC models.

    """)

    
    st.write("""
    ##### Artificial Neural Network:
    
    The optimal artificial neural network (ANN) evaluated in this study reflects a sequential structure with:

    - **One input layer** (12 input nodes)
    - **Three hidden layers** (24, 24, and 12 nodes, respectively)
    - **One single-node output layer**

    As the features are numerical real numbers within specific ranges, the choice of **'ReLU'** as the activation function for hidden layers is straightforward. Similarly, as this is a logistic classification task (where the output is binary: '0' for 'unstable' and '1' for 'stable'), the choice of **'sigmoid'** as the activation function for the output layer is appropriate.

    The model is compiled using:
    - **Optimizer**: 'adam'
    - **Loss function**: 'binary_crossentropy'
    - **Metric**: 'accuracy'

    Below is the ANN architecture visual representation:
    """)

    # Adding the image
    st.image('images/ann.png', caption="Artificial Neural Network Architecture", use_container_width=True)

    st.write("The app helps to make predictions about the stability of smart grids to better understand the system's behavior.")
