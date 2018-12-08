# install chrome
apt-get update -y
apt-get install -y libappindicator1 fonts-liberation
cd temp
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
dpkg -i google-chrome*.deb
apt-get -f -y install
dpkg --configure -a
rm google-chrome-stable_current_amd64.deb*

# take screenshot
#google-chrome-stable --headless --disable-gpu --screenshot --window-size=1280,1696 --no-sandbox file:///Users/edgartanaka/git/project-4/demofile.html