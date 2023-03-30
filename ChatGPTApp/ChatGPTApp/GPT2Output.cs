using Microsoft.ML.Data;

namespace ChatGPTApp;

public class GPT2Output
{
    [ColumnName("generated_text")]
    public string GeneratedText { get; set; }
}