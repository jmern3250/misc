#---------------------------------------
#	Import Libraries
#---------------------------------------
import clr
import sys
import json
import os
import random
import ctypes
import codecs
import time 

#---------------------------------------
#	[Required]	Script Information
#---------------------------------------
ScriptName = "Slots_v2"
Website = "http://www.google.com" #XXX Josh's website? 
Description = "Slot machine chatbot"
Creator = "MohnJern"
Version = "0.1.0"

#---------------------------------------
#	Set Variables
#---------------------------------------
configFile = "SlotsConfig.json"
settings = {}
responses = {}
game_settings ={}
emotes = []
slot1 = ""
slot2 = ""
slot3 = ""
user = ""
probs = [0.2, 0.1, .2/3., .04, .02] #1x, 2x, 3x, 5x, 10x


def ScriptToggled(state):
	return

#---------------------------------------
#	[Required] Intialize Data (Only called on Load)
#---------------------------------------
def Init():
	global responses, settings, configFile, emotes

	path = os.path.dirname(__file__)
	# try:
	# 	with codecs.open(os.path.join(path, configFile), encoding='utf-8-sig', mode='r') as file:
	# 		settings = json.load(file, encoding='utf-8-sig')
	# except:
	settings = dict(
			command_name='!slots',
			permission_level='Everyone',
			permission_level_info=None, #XXX
			usage='All',
			max_bet_amount=100, #XXX
			min_bet_amount=1, #XXX
			cooldown=60,
			user_cooldown=60,
			cooldown_response="$user, the command is still on cooldown for $cd seconds!",
			use_cooldown=True,
			)
	responses = dict(
				pull='$user pulls the lever...',
				lost='You lost, $user! Better luck next time!',  
				even='You broke even. Could be worse...',  
				won='You won $amt $currency $user! Good on you',
				jackpot='5X Jackpot, $user! You won $amt $currency! Spend it all in one place!',
				super_jackpot='10X SUPER JACKPOT!!! You won $amt $currency! $user is on top of the world now!',
				too_high='Bet amount over maximum, you high-roller you...',
				too_low="Bet amount below minimum - you're better than that",
				no_cash="Bet exceeds avaiable points! What are you trying to pull???"
				)
	game_settings = dict(
				superemote= 'KappaPride',
				emotelist=['Kappa', 'LUL', 'NotLikeThis', 'WutFace', 'MingLee'],
				multipliers=[1,2,3,5,10],
				houseedge=1.,
				)
	emotes.append(game_settings["superemote"])

	# responses.extend([settings["rewardTwoSame"], settings["rewardJackpot"], settings["rewardSuperJackpot"]])
	# try:
	# 	for i in responses:
	# 		int(i)
	# except:
	# 	MessageBox = ctypes.windll.user32.MessageBoxW
	# 	MessageBox(0, u"Invalid values", u"Slots Script failed to load. The rewards are not numbers.", 0)
	# return


#---------------------------------------
#	[Required] Execute Data / Process Messages
#---------------------------------------
def Execute(data):
	global probs, emotes, settings, userId, username, ScriptName

	if data.IsChatMessage() and data.GetParam(0).lower() == settings['command_name'] and Parent.HasPermission(data.User, settings['permission_level'], ""):
		ResponseStr = ""
		userId = data.User			
		username = data.UserName
		bet = None #XXX 

		if bet > > Parent.GetPoints(userId):
			ResponseStr = respones['no_cash'] 
		elif bet > settings['max_bet_amount']:
			ResponseStr = responses['too_high']
		elif bet < settings['min_bet_amount']:
			ResponseStr = responses['too_low']
		# Check if there is a cooldown active 
		elif settings["cooldown_active"] and (Parent.IsOnCooldown(ScriptName, settings["command_name"]) or Parent.IsOnUserCooldown(ScriptName, settings["command_name"], userId)):
			if Parent.GetCooldownDuration(ScriptName, settings["command_name"]) > Parent.GetUserCooldownDuration(ScriptName, settings["command_name"], userId):
				cd = Parent.GetCooldownDuration(ScriptName, settings["command_name"])
				ResponseStr = settings["cooldown_response"]
			else:
				cd = Parent.GetUserCooldownDuration(ScriptName, settings["command"], userId)
				ResponseStr = settings["cooldown_response"]
			ResponseStr = ResponseStr.replace("$cd", str(cd))
		else:
			Parent.RemovePoints(userId, username, bet)
			logit = random.uniform(0.,1.)
			logit *= game_settings['houseedge']
			outcome = 0
			ResponseStr = responses['lost']
			response_list = [responses['even'],
							 responses['won'],
							 responses['won'],
							 responses['jackpot'],
							 responses['super_jackpot']]
			for i, prob in enumerate(probs):
				if logit <= prob: 
					outcome = game_settings['multipliers'][i]
					ResponseStr = response_list[i]
			if outcome == 0:
				slots = []
				for _ in range(3):
					slots.append(str(random.choice(game_settings['multipliers'])))
			else:
				slots = [str(outcome)]*3

			Parent.SendStreamMessage(responses['pull'])
			for i, slot in enumerate(slots): 
				string = '|' + slot + 'X|'
				Parent.SendStreamMessage(string)
				time.sleep(2.1-i)
			
			reward = outcome*bet
			Parent.AddPoints(userId, username, int(reward))
			Parent.AddUserCooldown(ScriptName, settings["command_name"], userId, settings["user_cooldown"])
			Parent.AddCooldown(ScriptName, settings["command_name"], settings["cooldown"])

		ResponseStr = ResponseStr.replace("$amt", reward)
		ResponseStr = ResponseStr.replace("$user", username)
		ResponseStr = ResponseStr.replace("$currency", Parent.GetCurrencyName())

		Parent.SendStreamMessage(ResponseStr)
	return

#---------------------------------------
# Reload Settings on Save
#---------------------------------------
def ReloadSettings(jsonData):
	global responses, settings, configFile, emotes

	Init()

	twitchEmotes = set()
	# Grab the list of available Twitch emotes
	jsonData = json.loads(Parent.GetRequest("https://twitchemotes.com/api_cache/v3/global.json", {}))
	if jsonData["status"] == 200:
		twitchEmotes.update(set(json.loads(jsonData["response"]).keys()))

	# Grab the list of available Twitch emotes of the users channel
	jsonData = json.loads(Parent.GetRequest("https://decapi.me/twitch/subscriber_emotes/" + Parent.GetChannelName(), {}))
	if jsonData["status"] == 200:
		tempEmoteNames = jsonData["response"].split(" ")
		if tempEmoteNames[0] != "This": #channel has no sub button or sub button + no emotes
			twitchEmotes.update(set(tempEmoteNames))

	# Grab the list of available BetterTwitchTV emotes on the users channel
	jsonData = json.loads(Parent.GetRequest("https://api.betterttv.net/2/channels/" + Parent.GetChannelName(), {}))
	if jsonData["status"] == 200:
		for emote in json.loads(jsonData["response"])["emotes"]:
			twitchEmotes.add(emote['code'])

	# Grab the list of available global BetterTwitchTV emotes
	jsonData = json.loads(Parent.GetRequest("https://api.betterttv.net/2/emotes", {}))
	if jsonData["status"] == 200:
		for emote in json.loads(jsonData["response"])["emotes"]:
			twitchEmotes.add(emote['code'])

	invalidEmotes = []
	for emote in emotes:
		if emote not in twitchEmotes:
			invalidEmotes.append(emote)
	invalidEmotesString = ", ".join(invalidEmotes)

	if len(invalidEmotes) > 0:
		MessageBox = ctypes.windll.user32.MessageBoxW
		MessageBox(0, "Invalid Emotes: " + str(invalidEmotesString), u"Invalid Emote", 0)

	return

def OpenReadMe():
    location = os.path.join(os.path.dirname(__file__), "README.txt")
    os.startfile(location)
    return

#---------------------------------------
#	[Required] Tick Function
#---------------------------------------
def Tick():
	return
