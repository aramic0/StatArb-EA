//+------------------------------------------------------------------+
//| SA_Telegram.mqh — Standalone Telegram notifications for StatArb  |
//+------------------------------------------------------------------+
#ifndef SA_TELEGRAM_MQH
#define SA_TELEGRAM_MQH

input group "=============== Alerts & Telegram ==============="
input bool   send_alert            = false;  // Send Alert popup
input bool   send_notification     = false;  // Send Push notification
input bool   send_telegram_message = false;  // Send Telegram message
input string SA_TelegramBotToken   = "";     // Telegram Bot Token
input string SA_ChatId             = "";     // Telegram Chat ID
input string SA_TelegramApiUrl     = "https://api.telegram.org";  // Telegram API URL

// Aliases so StatArb.mqh SendTelegramMessage calls compile unchanged
#define TelegramApiUrl   SA_TelegramApiUrl
#define TelegramBotToken SA_TelegramBotToken
#define ChatId           SA_ChatId

//+------------------------------------------------------------------+
//| URL-encode text for Telegram API                                  |
//+------------------------------------------------------------------+
string SA_UrlEncode(string text)
{
    string result = "";
    uchar src[];
    StringToCharArray(text, src, 0, WHOLE_ARRAY, CP_UTF8);
    int len = ArraySize(src) - 1;

    for(int i = 0; i < len; i++)
    {
        uchar ch = src[i];
        if((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') ||
           (ch >= '0' && ch <= '9') || ch == '-' || ch == '_' || ch == '.' || ch == '~')
            result += CharToString((char)ch);
        else
            result += StringFormat("%%%02X", ch);
    }
    return result;
}

//+------------------------------------------------------------------+
//| Send message via Telegram (text only, no photo for standalone)   |
//+------------------------------------------------------------------+
bool SendTelegramMessage(string url, string token, string chat, string text,
                         string fileName = "")
{
    if(StringLen(token) == 0 || StringLen(chat) == 0) return false;

    string headers    = "";
    char   postData[];
    char   resultData[];
    string resultHeaders;
    int    timeout = 5000;

    ResetLastError();

    string encodedText = SA_UrlEncode(text);
    string requestUrl  = StringFormat("%s/bot%s/sendMessage?chat_id=%s&text=%s",
                                       url, token, chat, encodedText);

    int response = WebRequest("POST", requestUrl, headers, timeout,
                               postData, resultData, resultHeaders);

    if(response == 200)
    {
        Print("Telegram message sent successfully");
        return true;
    }
    else if(response == -1)
    {
        int err = GetLastError();
        Print("Telegram WebRequest error: ", err);
        if(err == 4014)
            PrintFormat("Add '%s' to allowed URLs in Tools > Options > Expert Advisors", url);
    }
    else
    {
        string result = CharArrayToString(resultData);
        PrintFormat("Telegram response %i: %s", response, result);
    }
    return false;
}

#endif // SA_TELEGRAM_MQH
