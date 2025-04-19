import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.3):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.redunce = nn.Linear(1280, hidden_size)

        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        redunce_embed =  self.redunce(input)
        embedded = self.dropout(redunce_embed)
        output, hidden = self.gru(embedded)
        return output, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.permute(0, 2, 1)

        weights = F.softmax(scores, dim=-1)
        if keys.dim() == 2:
            keys = keys.unsqueeze(0)
        context = torch.bmm(weights, keys)
        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.2):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.linear = nn.Linear(output_size, hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        #self.EOS_tensor = torch.tensor([1e-6, 1e-6, 1e-6, 1e-6, 1e-6], requires_grad=True).unsqueeze(0).unsqueeze(0)
        self.EOS_tensor = torch.tensor([0., 0., 0., 0., 1.], requires_grad=True, device=torch.device('cuda')).unsqueeze(0).unsqueeze(0)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = 1
        decoder_input = torch.zeros(batch_size, 5, dtype=torch.long).fill_(1e-6)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        decoder_outputs_ends = []
        attentions = []
        MAX_LENGTH=23

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            attentions.append(attn_weights)

            logits = decoder_output.squeeze(1)
            probs = F.softmax(logits, dim=-1)
            if i >= 5:
                if max(probs[0]) == probs[0][-1]:
                    for j in range(MAX_LENGTH - i):
                        decoder_outputs_ends.append(self.EOS_tensor)
                    break
            real_decoder_output = decoder_output[0][0].unsqueeze(0).unsqueeze(0)
            decoder_outputs.append(real_decoder_output)
            decoder_input = probs
        if len(decoder_outputs) == 0:
            decoder_outputs_finish = torch.cat(decoder_outputs_ends, dim=1)
        elif len(decoder_outputs_ends) == 0:
            decoder_outputs = torch.cat(decoder_outputs, dim=1)
            decoder_outputs_finish = F.softmax(decoder_outputs, dim=-1)
        else:
            decoder_outputs = torch.cat(decoder_outputs, dim=1)
            decoder_outputs = F.softmax(decoder_outputs, dim=-1)
            decoder_outputs_ends = torch.cat(decoder_outputs_ends, dim=1)
            decoder_outputs_finish = torch.cat((decoder_outputs, decoder_outputs_ends), dim=1)

        attentions = torch.cat(attentions, dim=1)
        return decoder_outputs_finish, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedding = self.linear(input.float().to(torch.device('cuda'))).float()
        embedding = embedding.unsqueeze(0)
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)

        input_gru = torch.cat((embedding, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)

        output = self.out(output)
        return output, hidden, attn_weights

class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):
        encoder_output, encoder_hidden = self.encoder(input)
        output, _, _ = self.decoder(encoder_output, encoder_hidden)
        return output