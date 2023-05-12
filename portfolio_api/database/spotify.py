from database.adatabase import ADatabase
import pandas as pd
class Spotify(ADatabase):
    def __init__(self):
        super().__init__("spotify")
    
    def find_included_playlists(self,track_uri):
        db = self.client["spotify"]
        table = db["songs"]
        data = table.find({"track_uri":track_uri},show_record_id=False)
        return pd.DataFrame(list(data))
    
    def find_playlist_info(self,pid):
        db = self.client["spotify"]
        table = db["playlists"]
        data = table.find({"pid":pid},show_record_id=False)
        return pd.DataFrame(list(data))
        
    def find_playlist_songs(self,pid):
        db = self.client["spotify"]
        table = db["songs"]
        data = table.find({"pid":pid},show_record_id=False)
        return pd.DataFrame(list(data))
    
    def find_song_uri(self,artist_name,track_name):
        db = self.client["spotify"]
        table = db["songs"]
        data = table.find({"artist_name":artist_name,"track_name":track_name},show_record_id=False)
        return pd.DataFrame(list(data))
