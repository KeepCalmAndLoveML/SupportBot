using System;
using System.Linq;
using System.Text;
using System.Timers;
using System.Diagnostics;
using System.Collections.Generic;


using Telegram.Bot;
using Telegram.Bot.Args;
using Telegram.Bot.Types;
using Telegram.Bot.Types.Enums;
using Telegram.Bot.Exceptions;

using F = System.IO.File;

namespace TeleBot
{
    public class SupportBot
    {
        const string AdminPassword = "GoToIsTheBest!";
        const string WelcomeMessage = @"Hi!
I am a bot
And I like cookies!
            ";

        readonly string PathToCsv;
        readonly string AccesToken;
        readonly string PythonFolder;
        readonly string PathToPythonBot;

        readonly int ResponsesPerQuestion;


        public bool SoMode { get; private set; }

        public TelegramBotClient Bot { get; set; }
        public User Me { get; set; }

        private int CountRatings { get; set; }
        private double AverageRating { get; set; }

        private Process Python { get; set; }
        private Timer RequestTimeout { get; set; }
        private Stack<long> Requests { get; set; }


        public SupportBot(string _accesToken, string _dbPath, string _bot, string _python, int _responses = 3, bool _so = false)
        {
            AccesToken = _accesToken;
            ResponsesPerQuestion = _responses;
            PathToCsv = _dbPath;
            PathToPythonBot = _bot;
            PythonFolder = _python;

            try
            {
                TestApiAsync();
            }
            catch (ApiRequestException)
            {
                throw new ArgumentException(string.Format("Invalid acces token: {0}", AccesToken));
            }

            Bot.OnUpdate += Respond;

            Python = new Process();

            Python.StartInfo = new ProcessStartInfo();
            Python.StartInfo.UseShellExecute = false;
            Python.StartInfo.FileName = "cmd.exe";
            Python.StartInfo.RedirectStandardInput = true;
            Python.StartInfo.RedirectStandardOutput = true;

            Python.Start();

            Python.StandardOutput.ReadLine();
            Python.StandardOutput.ReadLine();

            Python.StandardInput.WriteLine(string.Format(@"{0}\python.exe {1}", PythonFolder, PathToPythonBot));

            Python.StandardInput.WriteLine(PathToCsv);
            Python.StandardInput.WriteLine(ResponsesPerQuestion);

            Python.StandardOutput.ReadLine();
            Python.StandardOutput.ReadLine();

            CountRatings = 0;
            AverageRating = 0.0;

            SoMode = _so;

            RequestTimeout = new Timer();
            RequestTimeout.AutoReset = false;
            RequestTimeout.Enabled = false;
            RequestTimeout.Interval = 7000;
            RequestTimeout.Elapsed += RequestTimeout_Elapsed;

            Requests = new Stack<long>();
        }

        private async void RequestTimeout_Elapsed(object sender, ElapsedEventArgs e)
        {
            if (Requests.Count > 0)
            {
                long _id = Requests.Pop();
                await Bot.SendTextMessageAsync(_id, "Sorry, I couldn't find any similar question in my database");
            }
        }

        public void StartSpeaking()
        {
            if (!Bot.IsReceiving)
                Bot.StartReceiving();
        }

        private async void Respond(object sender, UpdateEventArgs e)
        {
            Update _received = e.Update;
            Console.WriteLine("Received Message!");
            if (_received.Type == UpdateType.MessageUpdate)
            {
                string text = _received.Message.Text;
                if (text == "/start")
                {
                    await Bot.SendTextMessageAsync(_received.Message.Chat.Id, WelcomeMessage);
                    return;
                }
                double _rate;
                if (double.TryParse(text, out _rate))
                {
                    await Bot.SendTextMessageAsync(_received.Message.Chat.Id, "Thank you for sharing your opinion!");
                    CountRatings++;
                    AverageRating = AverageRating + _rate / (double)CountRatings;

                    return;
                }
                if(text.StartsWith("/addqs " + AdminPassword))
                {
                    var _new = text.Replace(";!END!;", ",").Replace("\n", "");

                    try
                    {
                        F.AppendAllLines(PathToCsv, new string[] { _new });
                        await Bot.SendTextMessageAsync(_received.Message.Chat.Id, "Question added succesfully, restarting Bot");
                        try
                        {
                            RestartPython();
                            await Bot.SendTextMessageAsync(_received.Message.Chat.Id, "Restarted succesfully!");
                        }
                        catch
                        {
                            await Bot.SendTextMessageAsync(_received.Message.Chat.Id, "Unknown problem occured...");
                        }
                    }
                    catch
                    {
                        await Bot.SendTextMessageAsync(_received.Message.Chat.Id, "The was a problem while trying to add the question");
                    }

                    return;
                }
                else if (text.StartsWith("/addqs"))
                {
                    await Bot.SendTextMessageAsync(_received.Message.Chat.Id, "Wrong password!");

                    return;
                }
                if (text.StartsWith("/avgrate " + AdminPassword))
                {
                    await Bot.SendTextMessageAsync(_received.Message.Chat.Id, string.Format("Avg rating: {0}, Ratings: {1}", AverageRating, CountRatings));

                    return;
                }
                else if (text.StartsWith("/avgrate"))
                {
                    await Bot.SendTextMessageAsync(_received.Message.Chat.Id, "Wrong password!");

                    return;
                }
                await Bot.SendTextMessageAsync(_received.Message.Chat.Id, "Please wait a bit while I search in my database...");
                Python.StandardInput.WriteLine(text.Replace("\n", string.Empty));

                int count = 0;
                List<string> responses = new List<string>();

                string tmp;
                Requests.Push(_received.Message.Chat.Id);
                RequestTimeout.Start();
                while ((tmp = Python.StandardOutput.ReadLine()) != ";!END!;")
                {
                    if (RequestTimeout.Enabled)
                        RequestTimeout.Stop();

                    responses.Add(tmp);
                    count++;
                    if (count == ResponsesPerQuestion)
                        break;
                }

                if (responses.Count > 0)
                {
                    await Bot.SendTextMessageAsync(_received.Message.Chat.Id, "Here are some similar questions I found in my database:");
                    foreach (string element in responses)
                    {
                        string txt = element;
                        if (SoMode)
                            txt = "https://stackoverflow.com/questions/" + txt;

                        if (txt.Length > 0)
                            await Bot.SendTextMessageAsync(_received.Message.Chat.Id, txt);
                    }

                    await Bot.SendTextMessageAsync(_received.Message.Chat.Id, "Please give me a rating from 1 to 10 (1 - really bad, 10 - really cool)");
                }
            }
        }

        private async void TestApiAsync()
        {
            Bot = new TelegramBotClient(AccesToken);
            Me = await Bot.GetMeAsync();

            await Bot.SetWebhookAsync("");
        }
        
        private void RestartPython()
        {
            Python.StandardInput.WriteLine(";!END!;");

            Python.StandardOutput.ReadLine();
            Python.StandardInput.WriteLine(string.Format(@"{0}\python.exe {1}", PythonFolder, PathToPythonBot));
    
            Python.StandardInput.WriteLine(PathToCsv);
            Python.StandardInput.WriteLine(ResponsesPerQuestion);

            Python.StandardOutput.ReadLine();
            //Python.StandardOutput.ReadLine();
        }

        public static SupportBot FromSettings(string _path)
        {
            string[] lines = F.ReadAllLines(_path, Encoding.Unicode);
            try
            {
                string pyPath = string.Join(":", lines[0].Split(':').Skip(1));
                int responses = int.Parse(lines[1].Split(':')[1]);
                string db = string.Join(":", lines[2].Split(':').Skip(1));
                bool soMode;
                string tmp = lines[3].Split(':')[1];
                soMode = tmp.Equals("true", StringComparison.InvariantCultureIgnoreCase);
                string token = string.Join(":", lines[4].Split(':').Skip(1));
                string python = string.Join(":", lines[5].Split(':').Skip(1));

                return new SupportBot(token, db, pyPath, python, responses, soMode);
            }
            catch
            {
                throw new ArgumentException(string.Format("File at path {0} could not be read as a settings file", _path));
            }
        }
    }
}
