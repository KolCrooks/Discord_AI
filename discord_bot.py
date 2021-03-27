import json
import discord
from model_inferface import generateMessage


class DiscordClient(discord.Client):


    async def on_ready(self):
        message_output = []
        print('Logged on as {0}!'.format(self.user))
        self.temp = 0.01


    async def on_message(self, message):
        content = message.content.split(" ") 
        if content[0] == '!fake':
            starter = None
            if len(content) > 1:
                starter = " ".join(content[1:]) + " "
            await message.channel.send(generateMessage(starter=starter, temp = self.temp)[:-1])
        elif content[0] == "!t":
            self.temp = float(content[1])
            await message.channel.send(f'temp now {self.temp}')


client = DiscordClient()

with open('secrets.json') as secrets:
    token = json.load(secrets)['token']

client.run(token)