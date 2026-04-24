# How We Keep Our Machine Learning Model Healthy in Production

When we put a machine learning model into the real world as an API, things can change over time. The real-world data might start looking different from the data we used to train the model. When this happens, the model's performance can drop—this is often called "model rot" or "data drift."

To make sure our breast cancer prediction model stays accurate, we need a simple but solid plan to keep an eye on it. Here is our 4-step strategy to monitor the model's health.

### 1. Watch the Incoming Data (Data Drift)
Usually, trouble starts when the data coming into the API begins to look strange.
* **What we do:** We save the normal patterns (like the average values) of the 30 features from our training data. In the real world, we check the new incoming data against these normal patterns using basic math equations (like Z-scores).
* **Action:** If a feature, like "mean symmetry," suddenly looks way out of place compared to our training data, the system sends an automatic alert so the team can investigate.

### 2. Watch the Predictions (Output Drift)
Sometimes the incoming data looks fine, but for some reason, the model starts giving weird answers anyway.
* **What we do:** We look at the big picture of what the model is guessing. If it normally predicts 40% "Malignant" and 60% "Benign," but suddenly starts predicting 95% "Malignant," something is probably wrong.
* **Action:** If the model heavily favors one answer over the others (for example, making the same guess 80% of the time), we trigger a warning. This helps us catch silent bugs or big changes in who is using our app.

### 3. Check the Real Answers (Performance Tracking)
The absolute best way to know if our model is working is to compare its guesses with the real medical test results.
* **What we do:** We save every prediction our API makes. Later on, when the real medical results come back, we send those real answers back to our system using the `/feedback` endpoint.
* **Action:** The system calculates a running score of how accurate the recent predictions were. If the accuracy drops below our safety line (like 85%), an alert goes off to let our team know it's time to teach the model again using fresh data.

### 4. Check the Servers (System Health)
Not all problems are math problems; sometimes the computer server itself is struggling.
* **What we do:** We watch standard computer health signs, like how much memory the app uses and how fast it answers requests.
* **Action:** If the API gets too slow or runs out of memory, DevOps tools catch the problem. This makes sure the service always stays online and fast for everyone relying on it.
