addgroup music
adduser one
adduser two

# add a new user to group
useradd -g music guoxh
sudo passwd guoxh

# add a extied user to group
usermod -a -G music one

# made a folder
mkdir project_music

# change the group owner
chown :music project_music/
chmod 775 project_music/
