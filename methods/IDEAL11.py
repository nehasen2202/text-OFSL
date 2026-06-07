import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from methods.noisemeta_template import NoiseMetaTemplate
from sklearn.cluster import KMeans
import copy

class IDEAL(NoiseMetaTemplate):
    def __init__(self, model_func, n_way, n_support, image_size, outlier, ssl_feature_extractor,
                 image_files=None, image_labels=None, text_vectors=None,
                 eta=0.1, gamma=0.1, hidden_dim=20, noise_type='IT', attention_method='bilstm', device='cuda:0'):
        super(IDEAL, self).__init__(model_func,
                                    n_way,
                                    n_support,
                                    image_size=image_size,
                                    image_files=image_files,
                                    outlier=outlier,
                                    image_labels=image_labels,
                                    noise_type=noise_type,
                                    device=device)
        self.ssl_feature_extractor = copy.deepcopy(ssl_feature_extractor)
        self.loss_fn = nn.CrossEntropyLoss()
        self.attention_method = attention_method
        self.text_vectors = text_vectors  # Added for handling textual features
        
        #self.text_projection = nn.Linear(768, 1600).to(self.device)
        #self.combined_projection = nn.Linear(3200, 1600).to(self.device)
        
        # Dynamically set projection layers based on the feature extractor's output dimension.
        visual_dim = self.ssl_feature_extractor.final_feat_dim  # This will be 1600 for Conv4, 640 for ResNet12, etc.
        self.text_projection = nn.Linear(768, visual_dim).to(self.device)
        self.combined_projection = nn.Linear(2 * visual_dim, visual_dim).to(self.device)
        
        if self.attention_method == 'bilstm':
            self.corr_encoder = nn.LSTM(input_size=self.n_support, hidden_size=hidden_dim, num_layers=2,
                                        batch_first=True, bidirectional=True)
            self.norm_encoder = nn.Sequential(
                nn.Linear(2 * hidden_dim * self.n_support, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.n_support)
            )
        elif self.attention_method == 'transformer':
            self.attentive = nn.Transformer(d_model=self.n_support, nhead=5, dim_feedforward=2 * hidden_dim)
            self.norm_encoder = nn.Sequential(
                nn.Linear(self.n_support * self.n_support, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.n_support)
            )
        self.eta = eta
        self.gamma = gamma
        self.to(self.device)        
       
    def cosine_similarity(self, x1, x2):
        """Compute cosine similarity between x1 and x2."""
        x1_norm = F.normalize(x1, p=2, dim=-1)
        x2_norm = F.normalize(x2, p=2, dim=-1)
        return torch.mm(x1_norm, x2_norm.t())

    def compute_textual_similarity(self, text_query, text_proto):
        """Compute similarity using textual vectors."""
        return self.cosine_similarity(text_query, text_proto) * 10

    def get_attention_class(self, x):
        # Intra-class Calibration with textual features
        xx = x[:, :self.n_support].reshape(self.n_way * self.n_support, *x.size()[2:])
        z_support = self.ssl_feature_extractor.forward(xx).reshape(self.n_way, self.n_support, -1)
        z_support_norm = z_support / torch.norm(z_support, 2, 2).unsqueeze(2)  # [N, S, d]
        correlation = torch.bmm(z_support_norm, z_support_norm.permute(0, 2, 1))  # [N, S, S]
        
        if self.attention_method == 'bilstm':
            z_out, _ = self.corr_encoder.forward(correlation)  # [N, S, 2*hidden_dim]
        elif self.attention_method == 'transformer':
            z_out = self.attentive(correlation, correlation)  # [N, S, S]
        else:
            raise ValueError('Error attention method!')
        c_out = self.norm_encoder.forward(z_out.reshape(self.n_way, -1))  # [N, S]
        attention_class = F.softmax(c_out, dim=1)  # [N, S]
        return attention_class

    def get_attention_support(self, x):
        with torch.no_grad():
            xx = x[:, :self.n_support].reshape(self.n_way * self.n_support, *x.size()[2:])
            z_support = self.ssl_feature_extractor.forward(xx).reshape(self.n_way, self.n_support, -1)
            z_proto = z_support.mean(1)  # [N, visual_dim]
            z_proto = torch.nn.functional.normalize(z_proto, p=2, dim=1)  # Normalize visual prototypes

            text_proto = self.text_vectors[:, :self.n_support].mean(1)  # [N, text_dim]
            text_proto = torch.nn.functional.normalize(text_proto, p=2, dim=1)  # Normalize textual prototypes

            # Project text_proto to match visual feature dimensions
            text_proto_projected = self.text_projection(text_proto)  # [N, visual_dim]
            z_support_combined = torch.cat([z_proto, text_proto_projected], dim=1)  # [N, combined_dim]
            z_support_combined = torch.nn.functional.normalize(z_support_combined, p=2, dim=1)  # Normalize combined features

            combined_dim = z_support_combined.size(1)

            # Update combined projection layer dynamically if needed
            if self.combined_projection.in_features != combined_dim:
                self.combined_projection = nn.Linear(
                    combined_dim, self.ssl_feature_extractor.final_feat_dim
                ).to(self.device)

            # Project combined features back to match visual dimension
            z_support_combined_projected = self.combined_projection(z_support_combined)  # [N, visual_dim]
            z_support_combined_projected = torch.nn.functional.normalize(z_support_combined_projected, p=2, dim=1)

            # Reshape and normalize z_support for clustering
            z_support = z_support.reshape(-1, z_support.shape[-1])  # [N*S, visual_dim]
            z_support = torch.nn.functional.normalize(z_support, p=2, dim=1)  # Normalize z_support

            # Perform KMeans clustering
            cluster = KMeans(self.n_way, init=z_support_combined_projected.detach().cpu().numpy(), n_init=1)
            cluster.fit(z_support.cpu().numpy())

            # Extract cluster centers
            proto = torch.from_numpy(cluster.cluster_centers_).to(self.device)  # [N, visual_dim]

        sim = self.cosine_similarity(z_support, proto) * 10  # [N*S, N]
        attention_support = torch.softmax(sim, dim=0)  # [N*S, N]
        return attention_support




    def set_forward(self, x):
        attention_class = self.get_attention_class(x)
        attention_support = self.get_attention_support(x)

        # Visual feature extraction (scores_ssl)
        xx = x.reshape(self.n_way * (self.n_support + self.n_query), *x.size()[2:])
        z_all = self.ssl_feature_extractor.forward(xx)
        z_all = z_all.reshape(self.n_way, self.n_support + self.n_query, *z_all.shape[1:])  # [N, S+Q, d]
        z_support = z_all[:, :self.n_support]  # [N, S, d]
        z_query = z_all[:, self.n_support:]  # [N, Q, d]
        z_proto_class = torch.sum(z_support * attention_class.unsqueeze(2), dim=1)  # [N, d]
        z_proto_support = attention_support.transpose(0, 1) @ z_support.reshape(self.n_way * self.n_support, -1)
        z_proto = z_proto_class * 0.9 + z_proto_support * 0.1
        z_proto = torch.nn.functional.normalize(z_proto, p=2, dim=1)  # Normalize prototypes
        z_query = z_query.reshape(self.n_way * self.n_query, -1)  # [N*Q, d]
        z_query = torch.nn.functional.normalize(z_query, p=2, dim=1)  # Normalize queries
        scores_ssl = self.cosine_similarity(z_query, z_proto) * 10

        # Visual feature extraction (scores_sl)
        z_support, z_query = self.parse_feature(x)
        z_proto_class = torch.sum(z_support * attention_class.unsqueeze(2), dim=1)  # [N, d]
        z_proto_support = attention_support.transpose(0, 1) @ z_support.reshape(self.n_way * self.n_support, -1)
        z_proto = z_proto_class * 0.9 + z_proto_support * 0.1
        z_proto = torch.nn.functional.normalize(z_proto, p=2, dim=1)  # Normalize prototypes
        z_query = z_query.reshape(self.n_way * self.n_query, -1)  # [N*Q, d]
        z_query = torch.nn.functional.normalize(z_query, p=2, dim=1)  # Normalize queries
        scores_sl = self.cosine_similarity(z_query, z_proto) * 10

        # Textual feature extraction
        text_support = self.text_vectors[:, :self.n_support]  # [N, S, d_text]
        text_query = self.text_vectors[:, self.n_support:].reshape(self.n_way * self.n_query, -1)
        text_proto_class = torch.sum(text_support * attention_class.unsqueeze(2), dim=1)  # [N, d_text]
        text_proto_support = attention_support.transpose(0, 1) @ text_support.reshape(self.n_way * self.n_support, -1)
        text_proto = text_proto_class * 0.9 + text_proto_support * 0.1
        text_proto = torch.nn.functional.normalize(text_proto, p=2, dim=1)  # Normalize textual prototypes
        text_query = torch.nn.functional.normalize(text_query, p=2, dim=1)  # Normalize textual queries
        textual_scores = self.cosine_similarity(text_query, text_proto) * 10

        # Combine visual and textual scores
        combined_scores = scores_ssl + scores_sl + 0.3 * textual_scores
        return combined_scores, scores_ssl, scores_sl, textual_scores


    def set_forward_loss(self, x, y=None, noise_idxes=None):
        assert y is not None
        assert noise_idxes is not None
        # ------------------------Meta Loss------------------------
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).long()
        y_query = y_query.to(self.device)
        scores, _, _, _ = self.set_forward(x)
        loss_classification = self.loss_fn(scores, y_query)

        # ------------------------Intra-class Noise Loss------------------------
        attention_class = self.get_attention_class(x)
        mask = torch.zeros_like(attention_class).to(self.device)
        for i in range(self.n_way):
            mask[i, noise_idxes[i]] = 1
        loss_intra = torch.mean(torch.sum(-torch.log(
            torch.max(1 - attention_class, torch.tensor([1e-14]).to(self.device))) * mask, dim=1))

        # ------------------------Inter-class Noise Loss------------------------
        xx = x[:, :self.n_support].reshape(self.n_way * self.n_support, *x.size()[2:])
        yy = y[:, :self.n_support].reshape(-1)
        z = self.ssl_feature_extractor.forward(xx)  # [N*S, d]
        sim = self.cosine_similarity(z, z)
        mask = torch.ones_like(sim).to(self.device)
        for ii in range(z.shape[0]):
            mask[ii, yy[ii] == yy] = 1
            mask[ii, yy[ii] != yy] = -1
            mask[ii, ii] = 0
        loss_inter = -torch.mean(sim * mask)
        loss = loss_classification + loss_intra * self.eta + loss_inter * self.gamma
        return loss

    def train_loop(self, epoch, train_loader, optimizer):
        self.train()
        avg_loss = 0
        for i, (x, y, text_vector) in enumerate(train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            text_vector = text_vector.to(self.device)  # Move text_vector to the correct device

            # Update self.text_vectors for the current batch
            self.text_vectors = text_vector

            self.n_query = x.size(1) - self.n_support  # x:[N, S+Q, n_channel, h, w]
            self.n_way = x.size(0)
            
            x, y, noise_idxes = self.add_noise(x, y, noise_type=self.noise_type, aug=True, return_y=True)

            optimizer.zero_grad()
            loss = self.set_forward_loss(x, y, noise_idxes=noise_idxes)  # Use updated self.text_vectors internally
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            if self.verbose and (i % 10) == 0:
                print(f"Epoch {epoch} | Batch {i}/{len(train_loader)} | Loss: {avg_loss / (i + 1):.4f}")
        if not self.verbose:
            print(f"Epoch {epoch} | Loss: {avg_loss / len(train_loader):.4f}")
        return avg_loss / len(train_loader)


    def test_loop(self, test_loader, return_std=False, image_files=None, image_labels=None):
        self.eval()
        
        # Update image_files and image_labels if provided
        if image_files is not None:
            self.image_files = image_files
            self.image_labels = image_labels

        acc_all = []
        iter_num = len(test_loader)  # Total number of iterations

        with torch.no_grad():
            for i, (x, y, text_vector) in enumerate(test_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                text_vector = text_vector.to(self.device)  # Move text_vector to the correct device

                # Update self.text_vectors for the current batch
                self.text_vectors = text_vector

                self.n_query = x.size(1) - self.n_support  # x:[N, S+Q, n_channel, h, w]
                self.n_way = x.size(0)

                # Forward pass to get scores
                scores, _, _, _ = self.set_forward(x)  # Use updated self.text_vectors internally
                y_query = np.repeat(range(self.n_way), self.n_query)  # [0 0 0 1 1 1 ...]

                # Calculate accuracy
                topk_scores, topk_labels = scores.data.topk(1, 1, True, True)  # top1
                topk_ind = topk_labels.cpu().numpy()  # index of topk
                top1_correct = np.sum(topk_ind[:, 0] == y_query)

                acc_all.append(top1_correct / len(y_query) * 100)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)

        # Conditional printing based on verbose
        if self.verbose:
            print(f"{iter_num} Test Acc = {acc_mean:.2f}±{1.96 * acc_std / np.sqrt(iter_num):.2f}%")

        # Return accuracy values with or without std based on return_std flag
        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean


