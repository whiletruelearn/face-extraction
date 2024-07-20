from pytube import YouTube
from pytube.exceptions import AgeRestrictedError
import subprocess


class YoutubeVideoDownloader:
    """Download Youtube video from url"""
    def __init__(self, url:str , resolution: int ) -> None:
        self.url = url
        self.resolution = self._check_is_valid_resolution(resolution)

    def _check_is_valid_resolution(self,resolution):
        VALID_RESOLUTIONS = [320,720]
        if resolution not in VALID_RESOLUTIONS:
            raise ValueError(f"resolution should be one of {VALID_RESOLUTIONS}")

    @staticmethod
    def _generate_command(url: str):
        return f"youtube-dl {url} -f best"
    
    def download_video(self):
        try:
            yt = YouTube(self.url)
            filtered_streams = yt.streams.filter(res=f"{self.resolution}p", progressive=True)
            stream = filtered_streams.first()
            stream.download()
            
        except AgeRestrictedError as e:
            command = self._generate_command(self.url)
            subprocess.run(command, shell=True)
