cd /home/admin/python/MMEmo ; python3 train_text.py --module=basic_bert --dataset=iemocap-text-4 --train.batch_size=16 --from_bert_pretrain=True
cd /home/admin/python/MMEmo ; python3 train_text.py --module=basic_bert --dataset=iemocap-text-4 --train.batch_size=16 --from_bert_pretrain=False

cd /home/admin/python/MMEmo ; python3 train_speech.py --module=basic_bart --dataset=iemocap-audio-4 --from_bert_pretrain=False --padding-max-length=1200 --train.batch_size=16 --test.batch_size=16
cd /home/admin/python/MMEmo ; python3 train_speech.py --module=basic_bart --dataset=iemocap-audio-4 --from_bert_pretrain=False --padding-max-length=1200 --layers=6 --train.batch_size=16  --test.batch_size=16



cd /home/admin/python/MMEmo ; python3 train_ta.py --module=basic_bert --dataset=iemocap-ta-4 --train.batch_size=16 --padding-max-length=1200 --fusion_mode=concat
cd /home/admin/python/MMEmo ; python3 train_ta.py --module=basic_bert --dataset=iemocap-ta-4 --train.batch_size=16 --padding-max-length=1200 --fusion_mode=concat --from_bert_pretrain=False
cd /home/admin/python/MMEmo ; python3 train_ta.py --module=basic_bertp --dataset=iemocap-ta-4 --train.batch_size=16 --padding-max-length=1200 --fusion_mode=sum
cd /home/admin/python/MMEmo ; python3 train_ta.py --module=basic_bertp --dataset=iemocap-ta-4 --train.batch_size=16 --padding-max-length=1200 --fusion_mode=sum --from_bert_pretrain=False

