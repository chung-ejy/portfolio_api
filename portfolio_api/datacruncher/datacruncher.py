from database.adatabase import ADatabase
from database.spotify import Spotify
from modeler_strats.universal_modeler import UniversalModeler
from textblob import TextBlob
import pandas as pd
import pickle
import json
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
                        complete = Datacruncher.feedback_cruncher(data)
        return complete
    
    @classmethod
    def price_cruncher(self,data):
        try:
            project = data["project"]
            project_db = ADatabase(project.lower())
            project_db.connect()
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
        project_db.connect()
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
        project_db.connect()
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
        project_db.connect()
        project_db.store("data",pd.DataFrame([data]))
        project_db.disconnect()
        complete = {}
        return complete

    @classmethod
    def shuffle_cruncher(self,data):
        try:
            artist_name = data["artist_name"]
            track_name = data["track_name"]
            if len(artist_name) < 30 and len(track_name) < 30:
                spotify = Spotify()
                spotify.connect()
                current = spotify.find_song_uri(artist_name,track_name).iloc[0]
                spotify.store("data",pd.DataFrame([data]))
                spotify.disconnect()
                current_pid = current["pid"]
                uri = current["track_uri"]
                spotify.connect()
                included_playlists = spotify.find_included_playlists(uri)
                pids = included_playlists["pid"].unique()
                spotify.disconnect()
                aggregate = []
                spotify.connect()
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