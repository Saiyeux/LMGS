
# MonoGSʵʱ�������
monogs_live_config = {
    'Results': {
        'save_results': True,
        'use_gui': False,
        'eval_rendering': False,
        'use_wandb': False
    },
    'Dataset': {
        'type': 'realsense',  # ʹ��ʵʱ�������
        'sensor_type': 'monocular',
        'pcd_downsample': 64,
        'adaptive_pointsize': True
    },
    'Training': {
        'single_thread': True,
        'init_itr_num': 200,
        'tracking_itr_num': 30,
        'mapping_itr_num': 50
    }
}
