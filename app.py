import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models

# 🎛️ App Config
st.set_page_config("🤖 AI Playground | Interactive Neural Network Builder", layout="wide")
st.title("🎨 TensorFlow Neural Playground")
st.caption("🚀 Build, Train & Visualize Neural Networks in Real-Time – No Coding Needed!")

# ⚙️ Sidebar Configuration
with st.sidebar:
    st.header("🧰 Playground Controls")

    st.subheader("🎲 Dataset")
    dataset = st.selectbox("Choose your dataset", ["Moons 🌙", "Circles ⭕", "Classification 📊"])
    samples = st.slider("📦 Samples", 200, 5000, 1000, step=100)
    noise = st.slider("🌪️ Noise Level", 0.0, 1.0, 0.2)
    test_split = st.slider("🧪 Test Split (%)", 10, 50, 30) / 100

    st.subheader("🧠 Architecture")
    num_layers = st.slider("🔢 Hidden Layers", 1, 5, 2)
    layer_configs = []
    for i in range(num_layers):
        st.markdown(f"🔹 **Layer {i+1} Settings**")
        units = st.number_input(f"🔧 Units (Layer {i+1})", 1, 256, 8, key=f"units_{i}")
        activation = st.selectbox(f"⚡ Activation (Layer {i+1})", ["relu", "tanh", "sigmoid"], key=f"act_{i}")
        layer_configs.append((units, activation))

    st.subheader("🎯 Training")
    learning_rate = st.number_input("📉 Learning Rate", 0.0001, 1.0, 0.01)
    epochs = st.slider("⏳ Epochs", 100, 3000, 500, step=100)
    batch_size = st.slider("📦 Batch Size", 16, 256, 64)
    early_stop = st.checkbox("🛑 Early Stopping", True)
    patience = st.slider("⌛ Patience", 1, 20, 5) if early_stop else 0

# 🚀 Training Trigger
if st.button("🧠 Train Neural Network"):

    def generate_data():
        if dataset == "Moons 🌙":
            return make_moons(n_samples=samples, noise=noise, random_state=42)
        elif dataset == "Circles ⭕":
            return make_circles(n_samples=samples, noise=noise, factor=0.5, random_state=42)
        else:
            return make_classification(n_samples=samples, n_features=2, n_informative=2,
                                       n_redundant=0, n_clusters_per_class=1, random_state=42)

    # 📊 Data Preparation
    X, y = generate_data()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42)

    # 🏗️ Build Model
    model = models.Sequential()
    model.add(layers.Input(shape=(2,)))
    for units, activation in layer_configs:
        model.add(layers.Dense(units, activation=activation))
    model.add(layers.Dense(2, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 🧠 Callbacks
    callbacks = []
    if early_stop:
        callbacks.append(tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True))

    # 🌈 Setup Mesh Grid for Visualization
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # 📈 Training Loop with Visualization
    st.subheader("📈 Training in Progress...")
    progress_bar = st.progress(0)
    loss_chart = st.empty()
    decision_chart = st.empty()
    train_loss, val_loss = [], []

    for epoch in range(1, epochs + 1):
        history = model.fit(X_train, y_train,
                            epochs=1,
                            batch_size=batch_size,
                            verbose=0,
                            validation_data=(X_test, y_test),
                            callbacks=callbacks)

        train_loss.append(history.history["loss"][0])
        val_loss.append(history.history["val_loss"][0])

        if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            preds = np.argmax(model.predict(grid, verbose=0), axis=1).reshape(xx.shape)
            fig, ax = plt.subplots()
            ax.contourf(xx, yy, preds, alpha=0.8, cmap='Spectral')
            ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20, cmap='Spectral')
            ax.set_title(f"🌈 Decision Boundary @ Epoch {epoch}", fontsize=14)
            ax.set_xticks([]); ax.set_yticks([])
            decision_chart.pyplot(fig)

            fig2, ax2 = plt.subplots()
            ax2.plot(train_loss, label="Train Loss", color="blue")
            ax2.plot(val_loss, label="Val Loss", color="orange")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Loss")
            ax2.set_title("📉 Loss Curve")
            ax2.legend()
            loss_chart.pyplot(fig2)

        progress_bar.progress(epoch / epochs)

    # 🎓 Evaluation
    st.success("🏁 Training Complete!")
    y_train_pred = np.argmax(model.predict(X_train, verbose=0), axis=1)
    y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    st.metric("🎓 Train Accuracy", f"{accuracy_score(y_train, y_train_pred):.2%}")
    st.metric("📈 Test Accuracy", f"{accuracy_score(y_test, y_test_pred):.2%}")
    st.info("🔁 Want to experiment? Change settings from the sidebar and re-train your model!")
