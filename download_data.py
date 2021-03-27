import csv
import json
import discord

users_to_mimic = [280121047985160192, 245626897218797568, 289485232908795904, 272138230055567361, 250726400149946368, 214553950995611648, 304006547892076544, 260961292683051008]

class DiscordClient(discord.Client):
    async def on_ready(self):
        message_output = []
        print('Logged on as {0}!'.format(self.user))
        channel = self.get_channel(280140912754163712)
        messages = await channel.history(limit=100000).flatten()
        for message in messages:
            if message.author.id in users_to_mimic and ("https://" not in message.content) and message.content != "":
                message_output.append((message.id, message.content, message.author.name))
        with open('messages.csv', 'w', newline='', encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(message_output)
        print("Done!")


    async def on_message(self, message):
        print('Message from {0.author}: {0.content}'.format(message))

client = DiscordClient()

with open('secrets.json') as secrets:
    token = json.load(secrets)['token']

client.run(token)