from django.http.response import JsonResponse
import json
from django.views.decorators.csrf import csrf_exempt
# Create your views here.
from database.adatabase import ADatabase
from database.spotify import Spotify
from modeler_strats.universal_modeler import UniversalModeler

import pandas as pd
umod = UniversalModeler()

@csrf_exempt
def apiView(request):
    try:
        if request.method == "GET":
            project = request.GET.get("project")
            if project in ["Longshot","Comet"]:
                speculation_db = ADatabase(project.lower())
                speculation_db.cloud_connect()
                models = speculation_db.retrieve("models")
                speculation_db.disconnect()
                product = {}
                factors = [str(x) for x in range(14)]
                for i in range(14):
                    product[str(i)] = float(request.GET.get(f"{i}"))
                prediction_slice = pd.DataFrame([product])
                simulation = umod.recommend(models,prediction_slice.copy(),factors)
                simulation["prediction"] = (simulation["cat_prediction"] + simulation["xgb_prediction"] ) / 2
                product["prediction"] = round(simulation["prediction"].item(),2)
                product["project"] = project
                complete = product
            else:
                try:
                    artist_name = request.GET.get("artist_name")
                    track_name = request.GET.get("track_name")
                    spotify = Spotify()
                    spotify.cloud_connect()
                    current = spotify.find_song_uri(artist_name,track_name).iloc[0]
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
                except Exception as e:
                    complete = {"artist_name":artist_name,"track_name":track_name,"artist_rec":"none found","track_rec":"none found","error":str(e)}
        elif request.method == "DELETE":
            complete = {}
        elif request.method == "UPDATE":
            complete = {}
        elif request.method == "POST":
            complete = {}
        else:
            complete = {}
    except Exception as e:
        complete = {"data":[],"errors":str(e)}
    return JsonResponse(complete,safe=False)