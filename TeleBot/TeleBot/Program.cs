using System;
using System.IO;
using System.Reflection;

using System.Diagnostics;

namespace TeleBot
{
    class Program
    {
        static void Main(string[] args)
        {
            var currExe = Assembly.GetExecutingAssembly().Location;
            var currFolder = Path.GetDirectoryName(currExe);
            
            SupportBot MyBot = SupportBot.FromSettings(currFolder + @"\Settings.txt");
            MyBot.StartSpeaking();
           
            Console.ReadLine();
        }
    }
}
