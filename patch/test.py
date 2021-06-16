import torch
from torch.nn import CosineSimilarity

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Evaluator:
    def __init__(self, adv_mask_class) -> None:
        super().__init__()
        self.adv_mask_class = adv_mask_class

    def test(self):
        test_loss = 0.0
        all_test_image_clean = torch.empty(0, device=device)
        all_test_image_perturbed = torch.empty(0, device=device)
        with torch.no_grad():
            for img_batch, img_names in self.adv_mask_class.test_loader:
                (loss, _), others = self.adv_mask_class.forward_step(img_batch, self.adv_mask_class.best_patch, img_names)
                test_loss += loss.item()
                img_batch_applied = others[2]
                batch_clean_emb = self.adv_mask_class.embedder(img_batch.to(device))
                batch_per_emb = self.adv_mask_class.embedder(img_batch_applied)
                all_test_image_clean = torch.cat([all_test_image_clean, batch_clean_emb], dim=0)
                all_test_image_perturbed = torch.cat([all_test_image_perturbed, batch_per_emb], dim=0)

        test_loss = test_loss / len(self.adv_mask_class.test_loader)
        print('Test loss: {:.6}'.format(test_loss))

        clean_preds = (CosineSimilarity()(all_test_image_clean, self.adv_mask_class.target_embedding) > self.adv_mask_class.config.same_person_threshold)
        per_preds = (CosineSimilarity()(all_test_image_perturbed, self.adv_mask_class.target_embedding) > self.adv_mask_class.config.same_person_threshold)

        from sklearn.metrics import plot_precision_recall_curve
