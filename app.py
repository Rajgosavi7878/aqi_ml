'''
🚀 New Project Alert! 🌆

 I'm excited to share my latest creation: Indian City AQI Predictor Web App 🏙️

 🔗 Try it out here: [your-deployment-link]

💻 Tech Stack Used:

 ✅ Python – Core language for data processing and backend

 ✅ Flask – Lightweight web framework for serving predictions

 ✅ Scikit-learn – KMeans clustering to classify air quality

 ✅ Pickle – For saving the trained clustering model

 ✅ Tailwind CSS – Modern and clean UI styling

 ✅ HTML + Jinja – Templating for dynamic page rendering

 ✅ Matplotlib + Seaborn – Visualizing city-wise PM2.5 levels

🛠️ Features:

 1️⃣ Predicts PM2.5-based air quality category for any Indian city 🌍

 2️⃣ Trained a KMeans clustering model using real PM2.5 data of 50 cities

 3️⃣ Categories include: Good, Moderate, Unhealthy, Hazardous, etc.

 4️⃣ Provides health guidance based on air quality level 💡

 5️⃣ Transparent, blurred-glass UI with a background image for an immersive feel 🖼️

 6️⃣ User input is retained after prediction to enhance UX

 7️⃣ Fully dynamic backend prediction integrated with frontend forms 🔁

🙏 Huge thanks to who supported this project!

#Flask #Python #MachineLearning #TailwindCSS #DataScience #WebApp #AQIPredictor #KMeans #ScikitLearn #AirPollution #Jinja2


'''



# import the lib
import pandas as pd # to read and process the csv file
from flask import Flask,render_template,request # render_template used to render the HTML file
from pickle import load # load the model

# intialize the app
app = Flask(__name__)

# load the data
with open("aqi_model.pkl","rb") as f:
	model = load(f)

# load and preprocess city PM data
data = pd.read_csv("indian_city_pm25_data_50.csv")
grouped = data.groupby("city")["pm25"].mean().reset_index()
city_pm_dict = grouped.set_index("city")["pm25"].to_dict()

cluster_labels = {
    0: "Unhealthy",
    1: "Moderate",
    2: "Hazardous",
    3: "Unhealthy for Sensitive Groups",
    4: "Good",
    5: "Very Unhealthy"
}

health_guidance = {
    "Good": "Air quality is satisfactory, and air pollution poses little or no risk.",
    "Moderate": "Air quality is acceptable; however, there may be a concern for some people who are unusually sensitive.",
    "Unhealthy for Sensitive Groups": "Members of sensitive groups may experience health effects. The general public is less likely to be affected.",
    "Unhealthy": "Everyone may begin to experience health effects; members of sensitive groups may experience more serious effects.",
    "Very Unhealthy": "Health alert: everyone may experience more serious health effects.",
    "Hazardous": "Health warnings of emergency conditions. The entire population is more likely to be affected."
}

@app.route("/",methods=["GET","POST"])
def predict_aqi():
	result = ""
	city = ""
	pm_value = ""
	label = ""
	guidance = ""
	if request.method == "POST":
		city = request.form["city"].strip().title()	
		if city not in city_pm_dict:
			result = f"City '{city}' not found in data."
		else:
			pm_value = city_pm_dict[city]
			cluster = model.predict([[pm_value]])[0]
			label = cluster_labels.get(cluster, "Unknown")
			guidance = health_guidance.get(label, "")
			resul = f"{city}: PM2.5 = {pm_value}, Category = {label}"
			
	return render_template("home.html",city=city,pm25=pm_value,label=label,guidance=guidance,result=result)

if __name__ == "__main__":
	app.run(debug=True)
	