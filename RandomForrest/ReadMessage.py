from telethon import TelegramClient
from telethon.tl.functions.channels import GetFullChannelRequest
from telethon.tl.functions.channels import JoinChannelRequest
from telethon.tl.functions.contacts import ResolveUsernameRequest
from telethon.tl.functions.channels import GetMessagesRequest
from telethon.tl.functions.messages import GetDialogsRequest
from telethon.tl.types import InputPeerEmpty
from time import sleep
from telethon.utils import get_display_name
from pprint import pprint
#import pymongo
#from pymongo import MongoClient


api_id = 54908
api_hash = "d4d7552974c713fd19cad74dea82c414"
phone_number = '+989130757995'

client = TelegramClient('anon', api_id, api_hash)
assert client.connect()
if not client.is_user_authorized():
    client.send_code_request(phone_number)
    me = client.sign_in(phone_number, input('Enter code: '))


# dialogs = []
# users = []
# chats = []
# last_date = None
# chunk_size = 100

# result = client(GetDialogsRequest(
#     offset_date=last_date,
#     offset_id=0,
#     offset_peer=InputPeerEmpty(),
#     limit=chunk_size
# ))

# print("******************************************************************************************************************************************************************************************")

# print("result ok...\n")
# dialogs.extend(result.dialogs)
# print("dialogs ok...\n")
# users.extend(result.users)
# print("users ok...\n")
# chats.extend(result.chats)
# print("chats ok...\n")

# print(len(result.chats))

# print("******************************************************************************************************************************************************************************************")
# global politicalChannel
# global messageString
# global savedMassage
# savedMassage = 50000
# politicalChannel = [ "TahavolateMantaghe" , "IRTTir" , "fadayiian" , "goftan_nagid_admin" , "man_enghelabiam1" , "dolatebaharchannel" , "khatteemam" , "bbcpersian" , "dachstudien" , "RahbordChannel" , "anjoman_esf" , "farsivoa" , "president_iran" , "melatebidar" ]
# usualChannels = [ "felfel365" , "khanehkhorshid1" , "Havij" , "hanista_channel" , "niazcom_ir" , "Friedrich_nietzsche" , "AConfederacy_of_Dunces" , "livethelife" , "gallery_didar000" , "big_bangpage" , "cafe_ketab1"]


#mongoClient = MongoClient('localhost', 27017)
#telegramData = mongoClient.dataMining
#posts = telegramData.posts

# posts.drop()


# for i in range(0, len(result.chats)):
#     try:
#         if result.chats[i]:
#             print(result.chats[i])
#         print(i, ")", get_display_name(result.chats[i]), "\n")
#         print( "username= " , result.chats[i].username)
#         if result.chats[i].username in usualChannels:
#             print("This Channel Is Politic...")
#             Messages = client.get_message_history(result.chats[i], limit=5000 )  # limit=10
#             if savedMassage >= 50000 :
#                 break
#             for message in Messages:
#                 messageString = ""
#                 try:
#                     if message.media == None:
#                         print("usual message:")
#                         # print(message.message)
#                         messageString = message.message
#                     else:
#                         print("media message:")
#                         # print(message.media.caption)
#                         messageString = message.media.caption
#                         # if message.media != None:
#                         #     print(message)
#                         # else:
#                         #     print("Message Has Media !")
#                 except BaseException as e:
#                     print("webpage message:")
#                     print("Exception Handled: " , str(e))
#                     # print(message.message)
#                     messageString = message.message
#                 if messageString == None:
#                     print("Message Is None...")
#                 else:
#                     print(messageString)
#                     post = {"_id": savedMassage ,
#                             "text": messageString ,
#                             "class": 0
#                     }
#                     post_id = posts.insert_one(post).inserted_id
#                     print(post_id)
#                     savedMassage = savedMassage + 1
#                     if savedMassage >= 50000 :
#                         break
#                 print("______________")
#     except BaseException as e:
#         print("Exception Handled: " , str(e))
#     print("______________________________________________________________________")
#     # input()
# print("******************************************************************************************************************************************************************************************")
#


















