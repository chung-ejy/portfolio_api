from database.adatabase import ADatabase
from database.spotify import Spotify
from modeler_strats.universal_modeler import UniversalModeler
from textblob import TextBlob
import pandas as pd
import pickle
import json
from tensorflow.keras.models import model_from_json
from database.adatabase import ADatabase
from tensorflow.keras.preprocessing.sequence import pad_sequences
import base64
import pickle

umod = UniversalModeler()

class Datacruncher(object):
    @classmethod
    def factory(self,data):
        data = json.loads(data.decode("utf-8"))
        project = data["project"]
        if project in ["Longshot","Comet"]:
            complete = Datacruncher.price_cruncher(data)
        else:
            if project == "Faker":
                complete = Datacruncher.faker_cruncher(data)
            else:
                if project == "Shuffle":
                    complete = Datacruncher.shuffle_cruncher(data)
                else:
                    if project == "Dopa":
                        complete = Datacruncher.dopa_cruncher(data)
                    else:
                        if project == "Blog":
                            complete = Datacruncher.blog_cruncher(data)
                        else:
                            if project =="Reported":
                                complete = Datacruncher.reported_cruncher(data)
                            else:
                                complete = Datacruncher.feedback_cruncher(data)

        return complete
    
    @classmethod
    def reported_cruncher(self, data):
        # Extract user input from the data
        user_input = data["proompt"]

        # Step 1: Connect to the database and retrieve model and tokenizer
        db = ADatabase("reported")
        db.cloud_connect()
        model_df = db.retrieve("model")
        db.disconnect()

        # Step 2: Load the model architecture from JSON
        model_json = model_df["model"].item()  # Retrieve the model architecture JSON
        model = model_from_json(model_json)

        # Step 3: Deserialize and load the tokenizer
        tokenizer_encoded = model_df["tokenizer"].item()  # Retrieve the base64-encoded tokenizer
        tokenizer_serialized = base64.b64decode(tokenizer_encoded)
        tokenizer = pickle.loads(tokenizer_serialized)

        # Step 4: Deserialize and load the model weights
        weights_base64 = model_df["weights"].item()  # Retrieve the base64-encoded weights
        weights_serialized = base64.b64decode(weights_base64)
        model_weights = pickle.loads(weights_serialized)

        # Step 5: Set the deserialized weights to the model
        model.set_weights(model_weights)

        # Step 6: Tokenize and pad the user input sequence
        input_sequence = tokenizer.texts_to_sequences([user_input])

        # Check if the input sequence is empty after tokenization
        if len(input_sequence[0]) == 0:
            return {"response": "Invalid input: the input is not in the vocabulary."}

        # Step 7: Get the model's expected input length and pad the input sequence
        max_input_len = model.input_shape[1]  # Retrieve model's input length
        input_padded = pad_sequences(input_sequence, maxlen=max_input_len, padding='post')

        # Step 8: Predict the output sequence using the model
        predictions = model.predict(input_padded)

        # Step 9: Convert the prediction (token indices) back to text
        predicted_sequence = predictions.argmax(axis=-1)[0]  # Get the predicted token indices

        # Step 10: Create reverse mapping from token index to word
        reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

        # Convert token indices to words
        predicted_text = ' '.join([reverse_word_map.get(idx, '') for idx in predicted_sequence if idx != 0])

        return {"response": predicted_text}

    @classmethod
    def price_cruncher(self,data):
        try:
            project = data["project"]
            project_db = ADatabase(project.lower())
            project_db.cloud_connect()
            models = project_db.retrieve("models")
            project_db.store("data",pd.DataFrame([data]))
            project_db.disconnect()
            factors = [str(x) for x in range(14)]
            prediction_slice = pd.DataFrame([data])
            for factor in factors:
                prediction_slice[factor] = pd.to_numeric(prediction_slice[factor])
            simulation = umod.recommend(models,prediction_slice.copy(),factors)
            simulation["prediction"] = (simulation["cat_prediction"] + simulation["xgb_prediction"] ) / 2
            data["prediction"] = round(simulation["prediction"].item(),2)
            data["project"] = project
            complete = data
        except Exception as e:
            print(str(e))
            return {}
        return complete

    @classmethod
    def dopa_cruncher(self,data):
        project_db = ADatabase("dopa")
        factors = ["FirstBlood","FirstTower","FirstBaron","FirstDragon","FirstInhibitor"]
        project_db.cloud_connect()
        models = project_db.retrieve("models")
        project_db.store("data",pd.DataFrame([data]))
        project_db.disconnect()
        try:
            data["side"] = 1 if data["side"] else 0
            data["tier"] = "gm" if data["tier"] else "m"
            model = models[(models["tier"]==data["tier"]) & (models["side"]==data["side"])]
            complete = umod.recommend(model,pd.DataFrame([data]),factors).rename(columns={"xgb_prediction":"prediction"}).to_dict("records")[0]
            return complete
        except Exception as e:
            print(str(e))
            return {"msg":str(e)}
    
    @classmethod
    def faker_cruncher(self,data):
        project_db = ADatabase("news")
        project_db.cloud_connect()
        model = project_db.retrieve("models")
        project_db.store("data",pd.DataFrame([data]))
        project_db.disconnect()
        m = pickle.loads(model["model"].item())
        complete = {}
        texttb = TextBlob(data["text"])
        titletb = TextBlob(data["title"])
        complete["tpolarity"] = titletb.sentiment.polarity
        complete["tsubjectivity"] = titletb.sentiment.subjectivity
        complete["polarity"] = texttb.sentiment.polarity
        complete["subjectivity"] = texttb.sentiment.subjectivity
        classification = int(m.predict(pd.DataFrame([complete])))
        complete["classification"] = classification
        complete["title"] = data["title"]
        complete["text"] = data["text"]
        return complete
    
    @classmethod
    def feedback_cruncher(self,data):
        project_db = ADatabase("feedback")
        project_db.cloud_connect()
        project_db.store("data",pd.DataFrame([data]))
        project_db.disconnect()
        complete = {"project_name":"","user":"","feedback":"","project":"feedback"}
        return complete

    @classmethod
    def blog_cruncher(self):
        project_db = ADatabase("blogs")
        project_db.cloud_connect()
        blogs = project_db.retrieve("data")
        project_db.disconnect()
        blogs = blogs.to_dict("records")
        blogs.reverse()
        complete = {"blogs":blogs,"project":"blog"}
        return complete

    @classmethod
    def shuffle_cruncher(self,data):
        try:
            artist_name = data["artist_name"]
            track_name = data["track_name"]
            if len(artist_name) < 30 and len(track_name) < 30:
                spotify = Spotify()
                spotify.cloud_connect()
                current = spotify.find_song_uri(artist_name,track_name).iloc[0]
                spotify.store("data",pd.DataFrame([data]))
                spotify.disconnect()
                current_pid = current["pid"]
                uri = current["track_uri"]
                spotify.cloud_connect()
                included_playlists = spotify.find_included_playlists(uri)
                pids = included_playlists["pid"].unique()
                spotify.disconnect()
                aggregate = []
                spotify.cloud_connect()
                for pid in pids:
                    if pid != current_pid:
                        songs = spotify.find_playlist_songs(int(pid))
                        aggregate.append(songs)
                spotify.disconnect()
                s = pd.concat(aggregate)
                max_follower = s["num_holdouts"].max()
                s["follower_percentage"] = s["num_holdouts"] / max_follower
                s["count"] = 1 * s["follower_percentage"]
                analysis = s.groupby(["track_uri","artist_uri","artist_name","track_name"]).sum().reset_index()
                recs = analysis.sort_values("count",ascending=False)
                rec = recs[(recs["track_name"] != track_name)].sort_values("count",ascending=False).iloc[1]
                complete = {"artist_name":artist_name,"track_name":track_name,"artist_rec":rec["artist_name"],"track_rec":rec["track_name"]}
            else:
                complete = {"artist_name":artist_name,"track_name":track_name,"artist_rec":"none found","track_rec":"none found","error":str(e)}    
        except Exception as e:
            complete = {"artist_name":artist_name,"track_name":track_name,"artist_rec":"none found","track_rec":"none found","error":str(e)}
        return complete