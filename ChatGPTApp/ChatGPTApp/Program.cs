using Microsoft.ML;

namespace ChatGPTApp;

internal abstract class Program
{

    static string GenerateText(MLContext context, PredictionEngine<GPT2Input, GPT2Output> engine, string inputText)
    {
        var input = new GPT2Input { Text = inputText };
        var output = engine.Predict(input);
        return output.GeneratedText;
    }
    static void Main(string[] args)
    {
        var context = new MLContext();
        var transformerChain = context.Transforms.Text.TokenizeIntoWords("Tokens", "Text")
            .Append(context.Transforms.Conversion.MapValueToKey("TokenIds", "Tokens"))
            .Append(context.Transforms.Text.ProduceNgrams("Ngrams", "TokenIds"))
            .Append(context.Transforms.Concatenate("Features", "Ngrams"))
            .Append(context.Transforms.NormalizeLpNorm("Features"))
            .AppendCacheCheckpoint(context);
        
        const string modelPath = "/Users/rminafa/Projects/DotNet_ChatGpt_Application/ChatGPTApp/ChatGPTApp/gpt-2-master/src/model.py";
        var engine = context.Model.CreatePredictionEngine<GPT2Input, GPT2Output>(transformerChain.Fit(context.Data.LoadFromEnumerable(new List<GPT2Input> { new GPT2Input { Text = "Hello" } })));
        var text = GenerateText(context, engine, "Hello");
        
        // use the model to make a prediction
        var prediction = engine.Predict(new GPT2Input { Text = "Hello" });
        
        Console.WriteLine($"Text: {prediction.GeneratedText}"); // Text: Hello
        
        
        Console.WriteLine("Press any key to exit...");
        
        Console.ReadKey();

    }
}