# linkedin_apply
Small bot that applies to LinkedIn Easy Apply jobs for you. Not really a choosy picker, will pretty much apply to anything it can get it's hands on (which is honestly what I'm doing in this job market).

# What it needs to work
- Windows (for now)
- Chromedriver downloaded and stored in your working directory <br>
These can be found at: http://chromedriver.chromium.org/downloads
- Python 3.x
- Selenium and BeautifulSoup4 imports: <br>
If you have pip, these can be downloaded by the commands
> pip install selenium </br>
> pip install BeautifulSoup4
- You must have filled out a LinkedIn easy apply application before (LinkedIn stores your answers so you don't have to fill them again or
think about them again)

# (Possible) Bugs:
- Does NOT have functionality to apply through third-party application websites (i.e., anything not easy apply)
- Does NOT have functionality to selectively choose jobs (apart from LinkedIn's own filters)
- Does this weird thing where it opens first 0 to k jobs on the page, then 0 to m jobs (k > m), then
 0 to n jobs (n > m) until the loop ends (the end of the loop is an arbitrary counter for now)
- Breaks if something unexpected happens (eg. a job that does not have the preset questions even for linkedIn easy apply)

Thank you! <br>
For comments, compliments, or constructive criticism please email ayushlall@g.ucla.edu
# linkedin_bot
# epl_data_analysis
