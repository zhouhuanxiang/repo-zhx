function evaluate()
%  vimeo_test('/home/web_server/zhouhuanxiang/disk/log/results/EDVR_GCB_Vimeo_woTSA_M/Vimeo', '/home/web_server/zhouhuanxiang/disk/vimeo/vimeo_septuplet/sequences', 'blocky')
  vimeo_test('/home/web_server/zhouhuanxiang/disk/log/results/EDVR_Vimeo_woTSA_M/Vimeo', '/home/web_server/zhouhuanxiang/disk/vimeo/vimeo_septuplet/sequences', 'blocky')
  % vimeo_test('/home/web_server/zhouhuanxiang/disk/vimeo/vimeo_septuplet/sequences_blocky37', '/home/web_server/zhouhuanxiang/disk/vimeo/vimeo_septuplet/sequences', 'blocky')
end

function result = vimeo_test(output_dir, target_dir, task)
  ts = get_task(output_dir, target_dir, task);
  [p, s, a] = run_eval_template(ts{1,2},ts{1,3},ts{1,4},get_path(ts{1,5}),ts{1,6},ts{1,7});
  result = [ts{1,1} ' psnr,ssim,abs= ' num2str(p) ', ' num2str(s) ', ' num2str(a)];
  result
end

% /home/web_server/MATLAB/R2017b/bin/matlab -nodisplay -nosplash -nodesktop -r "run('/home/web_server/zhouhuanxiang/mmsr/codes/evaluation/evaluate.m');quit;"