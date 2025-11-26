#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <getopt.h>

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "rknn_api.h"
#include "rkllm.h"

typedef struct {
	rknn_context rknn_ctx;
	rknn_input_output_num io_num;
	int model_channel;
	int model_width;
	int model_height;
	int model_image_token;
	int model_embed_size;
	rknn_tensor_attr *input_attrs;
	rknn_tensor_attr *output_attrs;
	rknn_input *inputs;
	rknn_output *outputs;
} rknn_app_context_t;

static void rknn_tensor_dump_attrs(rknn_tensor_attr *attr) {
	printf("  index=%d, name=%s, n_dims=%d, dims=[", attr->index, attr->name, attr->n_dims);
	for (int i = 0; i < attr->n_dims; i++) {
		printf("%d", attr->dims[i]);
		if (i + 1 != attr->n_dims)
			putc(',', stdout);
	}
	printf("], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, zp=%d, scale=%f\n",
		   attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
		   get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static int imgenc_init(const char *model_path, rknn_app_context_t *app_ctx, const int core_num) {
	int ret;

	ret = rknn_init(&app_ctx->rknn_ctx, (void *)model_path, 0, 0, NULL);
	if (ret < 0) {
		printf("rknn_init fail! ret=%d\n", ret);
		return -1;
	}

	printf("NPU core numbers is %d\n", core_num);
	//  在此设置多核推理
	if (core_num == 2)
		ret = rknn_set_core_mask(app_ctx->rknn_ctx, RKNN_NPU_CORE_0_1);
	else if (core_num == 3)
		ret = rknn_set_core_mask(app_ctx->rknn_ctx, RKNN_NPU_CORE_0_1_2);
	else
		ret = rknn_set_core_mask(app_ctx->rknn_ctx, RKNN_NPU_CORE_AUTO);
	if (ret < 0) {
		printf("rknn_set_core_mask fail! ret=%d\n", ret);
		return -1;
	}

	// Get Model Input Output Number
	ret = rknn_query(app_ctx->rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &app_ctx->io_num, sizeof(app_ctx->io_num));
	if (ret != RKNN_SUCC) {
		printf("rknn_query fail! ret=%d\n", ret);
		return -1;
	}

	// Get Model Input Info
	printf("model input tensors:\n");
	app_ctx->input_attrs = (rknn_tensor_attr *)calloc(app_ctx->io_num.n_input, sizeof(*app_ctx->input_attrs));
	for (int i = 0; i < app_ctx->io_num.n_input; i++) {
		app_ctx->input_attrs[i].index = i;
		ret = rknn_query(app_ctx->rknn_ctx, RKNN_QUERY_INPUT_ATTR,
						 &(app_ctx->input_attrs[i]), sizeof(*app_ctx->input_attrs));
		if (ret != RKNN_SUCC) {
			printf("rknn_query fail! ret=%d\n", ret);
			return -1;
		}
		rknn_tensor_dump_attrs(&(app_ctx->input_attrs[i]));
	}
	if (app_ctx->input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
		app_ctx->model_channel = app_ctx->input_attrs[0].dims[1];
		app_ctx->model_height = app_ctx->input_attrs[0].dims[2];
		app_ctx->model_width = app_ctx->input_attrs[0].dims[3];
	} else {
		app_ctx->model_height = app_ctx->input_attrs[0].dims[1];
		app_ctx->model_width = app_ctx->input_attrs[0].dims[2];
		app_ctx->model_channel = app_ctx->input_attrs[0].dims[3];
	}

	// Get Model Output Info
	printf("model output tensors:\n");
	app_ctx->output_attrs = (rknn_tensor_attr *)calloc(app_ctx->io_num.n_output, sizeof(*app_ctx->output_attrs));
	for (int i = 0; i < app_ctx->io_num.n_output; i++) {
		app_ctx->output_attrs[i].index = i;
		ret = rknn_query(app_ctx->rknn_ctx, RKNN_QUERY_OUTPUT_ATTR,
						 &(app_ctx->output_attrs[i]), sizeof(*app_ctx->output_attrs));
		if (ret != RKNN_SUCC) {
			printf("rknn_query fail! ret=%d\n", ret);
			return -1;
		}
		rknn_tensor_dump_attrs(&(app_ctx->output_attrs[i]));
	}
	app_ctx->model_image_token = app_ctx->output_attrs[0].dims[app_ctx->output_attrs[0].n_dims - 2];
	app_ctx->model_embed_size = app_ctx->output_attrs[0].dims[app_ctx->output_attrs[0].n_dims - 1];

	printf("model input format=%s, height=%d, width=%d, channel=%d; output token=%d, embed_size=%u\n",
		   app_ctx->input_attrs[0].fmt == RKNN_TENSOR_NCHW ? "NCHW" : "NHWC",
		   app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel,
		   app_ctx->model_image_token, app_ctx->model_embed_size);

	app_ctx->inputs = (rknn_input *)malloc(app_ctx->io_num.n_input * sizeof(*app_ctx->inputs));
	app_ctx->outputs = (rknn_output *)malloc(app_ctx->io_num.n_output * sizeof(*app_ctx->outputs));

	return 0;
}

static int imgenc_release(rknn_app_context_t *app_ctx) {
	if (app_ctx->inputs != NULL) {
		free(app_ctx->inputs);
		app_ctx->inputs = NULL;
	}
	if (app_ctx->outputs != NULL) {
		free(app_ctx->outputs);
		app_ctx->outputs = NULL;
	}
	if (app_ctx->input_attrs != NULL) {
		free(app_ctx->input_attrs);
		app_ctx->input_attrs = NULL;
	}
	if (app_ctx->output_attrs != NULL) {
		free(app_ctx->output_attrs);
		app_ctx->output_attrs = NULL;
	}
	if (app_ctx->rknn_ctx != 0) {
		rknn_destroy(app_ctx->rknn_ctx);
		app_ctx->rknn_ctx = 0;
	}
	return 0;
}

static int imgenc_run(rknn_app_context_t *app_ctx, void *img_data, float *out_result) {
	int ret;

	memset(app_ctx->inputs, 0, app_ctx->io_num.n_input * sizeof(*app_ctx->inputs));
	memset(app_ctx->outputs, 0, app_ctx->io_num.n_output * sizeof(*app_ctx->outputs));

	// Set Input Data
	app_ctx->inputs[0].index = 0;
	app_ctx->inputs[0].type = RKNN_TENSOR_UINT8;
	app_ctx->inputs[0].fmt = RKNN_TENSOR_NHWC;
	app_ctx->inputs[0].size = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
	app_ctx->inputs[0].buf = img_data;

	ret = rknn_inputs_set(app_ctx->rknn_ctx, 1, app_ctx->inputs);
	if (ret < 0) {
		printf("rknn_input_set fail! ret=%d\n", ret);
		return -1;
	}

	// Run
	ret = rknn_run(app_ctx->rknn_ctx, nullptr);
	if (ret < 0) {
		printf("rknn_run fail! ret=%d\n", ret);
		return -1;
	}

	// Get Output
	app_ctx->outputs[0].want_float = 1;
	ret = rknn_outputs_get(app_ctx->rknn_ctx, 1, app_ctx->outputs, NULL);
	if (ret < 0) {
		printf("rknn_outputs_get fail! ret=%d\n", ret);
	} else {
		// Post Process
		memcpy(out_result, app_ctx->outputs[0].buf, app_ctx->outputs[0].size);

		// Remeber to release rknn output
		rknn_outputs_release(app_ctx->rknn_ctx, 1, app_ctx->outputs);
	}

	return ret;
}

static uint64_t get_timestamp() {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ((uint64_t)ts.tv_sec) * 1000000 + ts.tv_nsec / 1000;
}

static const char *mtmd_system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
static const char *mtmd_prompt_prefix = "<|im_start|>user\n";
static const char *mtmd_prompt_postfix = "<|im_end|>\n<|im_start|>assistant\n";
static int save_image = 0;
static int save_embed = 0;
static LLMHandle llmHandle = nullptr;

static const struct option long_options[] = {
	{"usage", no_argument, NULL, '?'},
	{"core_num", required_argument, NULL, 'o'},
	{"max_context_len", required_argument, NULL, 'l'},
	{"max_new_tokens", required_argument, NULL, 'n'},
	{"chat_template", optional_argument, NULL, 't'},
	{"img_token", required_argument, NULL, 'm'},
	{"img_size", required_argument, NULL, 's'},
	{"save_image", no_argument, &save_image, 1},
	{"save_embed", no_argument, &save_embed, 1},
	{NULL, 0, NULL, 0}};

static void usage(const char *name) {
	printf("Usage: %s [options] image_path encoder_model_path llm_model_path\n", name);
	puts("  --core_num          NPU core number: 2 for rk3588, 2 for rt3576, 1 for others\n"
		 "  --max_context_len   max of total context length, default is 4095\n"
		 "  --max_new_tokens    max tokens the model will generate, default is 255\n"
		 "  --chat_template     chat template file if rkllm can't retrieve from model\n"
		 "  --img_tokens        default is <|vision_start|>,<|vision_end|>,<|image_pad|>\n"
		 "  --img_size          optional image size as format Height,Weight");
	exit(0);
}

static void exit_handler(int signal) {
	if (llmHandle != nullptr) {
		puts("Exiting");
		LLMHandle _tmp = llmHandle;
		llmHandle = nullptr;
		rkllm_destroy(_tmp);
	}
	exit(signal);
}

static int callback(RKLLMResult *result, void *userdata, LLMCallState state) {
	if (state == RKLLM_RUN_FINISH)
		printf("\n");
	else if (state == RKLLM_RUN_ERROR)
		printf("run error\n");
	else if (state == RKLLM_RUN_NORMAL)
		printf("%s", result->text);
	return 0;
}

static cv::Mat img_load(const char *image_path, const int image_width, const int image_height) {
	// The image is read in BGR format
	cv::Mat img = cv::imread(image_path);
	printf("load image file: %dx%d\n", img.cols, img.rows);
	// convert to RGB
	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

	double xs = (double)image_width / (double)img.cols;
	double ys = (double)image_height / (double)img.rows;

	int x, y, width, height;
	if (xs <= ys) {
		x = 0;
		width = image_width;
		height = (int)std::round(img.rows * xs);
		if (height > image_height)
			height = image_height;
		y = (image_height - height) / 2;
	} else {
		width = (int)std::round(img.cols * ys);
		if (width > image_width)
			width = image_width;
		x = (image_width - width) / 2;
		y = 0;
		height = image_height;
	}

	printf("scale image to %dx%d at offset (%d, %d)\n", width, height, x, y);
	cv::Mat input_img = cv::Mat(cv::Size(image_width, image_height), CV_8UC3, cv::Scalar(127, 127, 127));
	cv::Mat scaled_img = cv::Mat(input_img, cv::Rect(x, y, width, height));
	cv::resize(img, scaled_img, cv::Size(scaled_img.cols, scaled_img.rows));
	if (save_image)
		cv::imwrite("image.jpg", input_img);

	return input_img;
}

using namespace std;

int main(int argc, char **argv) {
	const char *chat_template = NULL;
	const char *system_prompt;
	const char *prompt_prefix;
	const char *prompt_postfix;
	int core_num = 2;
	int model_image_token = 0;
	int image_height = 0;
	int image_width = 0;

	RKLLMParam param = rkllm_createDefaultParam();
	param.top_k = 1;
	param.max_context_len = 4096;
	param.max_new_tokens = 256;
	param.skip_special_token = true;
	param.extend_param.base_domain_id = 1;
	param.img_start = "<|vision_start|>";
	param.img_end = "<|vision_end|>";
	param.img_content = "<|image_pad|>";

	int argv_off = 1;
	while (true) {
		/* getopt_long stores the option index here. */
		int option_index = 0;
		int c = getopt_long(argc, argv, "", long_options, &option_index);

		/* Detect the end of the options. */
		if (c == -1)
			break;

		argv_off++;
		if (c == 0)
			continue;

		switch (c) {
		case 'o':
			core_num = strtol(optarg, NULL, 10);
			break;
		case 'n':
			param.max_new_tokens = strtol(optarg, NULL, 10);
			break;
		case 'l':
			param.max_context_len = strtol(optarg, NULL, 10);
			break;
		case 't':
			chat_template = optarg != NULL ? optarg : "";
			break;
		case 'm': {
			char *ch1 = strchr(optarg, ',');
			if (ch1 == NULL)
				usage(argv[0]);
			param.img_start = strndup(optarg, ch1 - optarg);
			ch1++;
			char *ch2 = strchr(ch1, ',');
			if (ch2 == NULL)
				usage(argv[0]);
			param.img_end = strndup(ch1, ch2 - ch1);
			param.img_content = strdup(ch2 + 1);
			break;
		}
		case 's': {
			char *ch1 = strchr(optarg, ',');
			if (ch1 == NULL) {
				image_height = (int)strtol(optarg, NULL, 10);
				image_width = image_height;
			} else {
				char num[ch1 - optarg + 1];
				strncpy(num, optarg, sizeof(num));
				image_height = (int)strtol(num, NULL, 10);
				image_width = (int)strtol(ch1 + 1, NULL, 10);
			}
			break;
		}
		case '?':
		default:
			usage(argv[0]);
		}
	}
	puts("args:");
	for (int i = argv_off; i < argc; i++)
		printf("  %s\n", argv[i]);

	if (argc < argv_off + 3)
		usage(argv[0]);

	const char *image_path = argv[argv_off];
	const char *encoder_model_path = argv[argv_off + 1];
	param.model_path = argv[argv_off + 2];

	if (chat_template != NULL) {
		if (strlen(chat_template) == 0) {
			system_prompt = mtmd_system_prompt;
			prompt_prefix = mtmd_prompt_prefix;
			prompt_postfix = mtmd_prompt_postfix;
		} else {
			puts("FIXME: need to parse template file");
			exit(1);
		}
	}

	printf("Tokens: img_start_token=%s, img_end_token=%s, img_content_token=%s\n",
		   param.img_start, param.img_end, param.img_content);

	int ret;
	uint64_t t_start_us, t_end_us;

	rknn_app_context_t rknn_app_ctx;
	memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
	t_start_us = get_timestamp();
	ret = imgenc_init(encoder_model_path, &rknn_app_ctx, core_num);
	if (ret != 0) {
		printf("imgenc_init fail! ret=%d model_path=%s\n", ret, encoder_model_path);
		exit_handler(-1);
	}
	t_end_us = get_timestamp();
	printf("ImgEnc Model loaded in %.2f ms\n", (t_end_us - t_start_us) / 1000.0);

	if (rknn_app_ctx.model_width != 0 && rknn_app_ctx.model_height != 0) {
		image_height = rknn_app_ctx.model_height;
		image_width = rknn_app_ctx.model_width;
	} else {
		if (image_width == 0 || image_height == 0) {
			puts("[Error] Please specify image height and width for model input");
			exit_handler(-1);
		}
		rknn_app_ctx.model_height = image_height;
		rknn_app_ctx.model_width = image_width;
	}

	t_start_us = get_timestamp();
	cv::Mat input_img = img_load(image_path, image_width, image_height);
	t_end_us = get_timestamp();
	printf("Load and preprocess the image cost %.2f ms\n", (t_end_us - t_start_us) / 1000.0);

	model_image_token = rknn_app_ctx.model_image_token;
	float img_vec[model_image_token * rknn_app_ctx.model_embed_size];

	t_start_us = get_timestamp();
	ret = imgenc_run(&rknn_app_ctx, input_img.data, img_vec);
	if (ret != 0) {
		printf("imgenc_run fail! ret=%d\n", ret);
		exit_handler(-1);
	}
	t_end_us = get_timestamp();
	printf("Encode the image cost %.2f ms\n", (t_end_us - t_start_us) / 1000.0);

	ret = imgenc_release(&rknn_app_ctx);
	if (ret != 0)
		printf("imgenc_release fail! ret=%d\n", ret);

	if (save_embed) {
		FILE *f = fopen("img_vec.bin", "wb");
		if (f != NULL) {
			fwrite(img_vec, sizeof(img_vec), 1, f);
			fclose(f);
		}
	}

	t_start_us = get_timestamp();
	ret = rkllm_init(&llmHandle, &param, callback);
	if (ret == 0) {
		printf("rkllm init success\n");
	} else {
		printf("rkllm init failed\n");
		exit_handler(-1);
	}
	t_end_us = get_timestamp();
	printf("LLM Model loaded in %.2f ms\n", (t_end_us - t_start_us) / 1000.0);

	RKLLMInput rkllm_input;
	RKLLMInferParam rkllm_infer_params;

	memset(&rkllm_input, 0, sizeof(RKLLMInput));
	memset(&rkllm_infer_params, 0, sizeof(RKLLMInferParam));

	rkllm_infer_params.mode = RKLLM_INFER_GENERATE;
	rkllm_infer_params.keep_history = 0;
	if (chat_template != NULL)
		rkllm_set_chat_template(llmHandle, system_prompt, prompt_prefix, prompt_postfix);

	vector<string> pre_input;
	pre_input.push_back("<image>What is in the image?");
	pre_input.push_back("<image>这张图片中有什么？");
	puts("\n****************** 可输入以下问题对应序号获取回答/或自定义输入 ****************");
	for (int i = 0; i < (int)pre_input.size(); i++)
		printf("[%d] %s\n", i, pre_input[i].c_str());
	puts("*************************************************************************");

	while (true) {
		std::string input_str;
		printf("\n");
		printf("user: ");
		std::getline(std::cin, input_str);
		if (input_str == "exit")
			break;
		if (input_str == "clear") {
			ret = rkllm_clear_kv_cache(llmHandle, 1, nullptr, nullptr);
			if (ret != 0)
				printf("clear kv cache failed!\n");
			continue;
		}
		for (int i = 0; i < (int)pre_input.size(); i++) {
			if (input_str == to_string(i)) {
				input_str = pre_input[i];
				cout << input_str << endl;
			}
		}
		if (input_str.find("<image>") == std::string::npos) {
			rkllm_input.input_type = RKLLM_INPUT_PROMPT;
			rkllm_input.role = "user";
			rkllm_input.prompt_input = (char *)input_str.c_str();
		} else {
			rkllm_input.input_type = RKLLM_INPUT_MULTIMODAL;
			rkllm_input.role = "user";
			rkllm_input.multimodal_input.prompt = (char *)input_str.c_str();
			rkllm_input.multimodal_input.image_embed = img_vec;
			rkllm_input.multimodal_input.n_image_tokens = model_image_token;
			rkllm_input.multimodal_input.n_image = 1;
			rkllm_input.multimodal_input.image_height = image_height;
			rkllm_input.multimodal_input.image_width = image_width;
		}
		printf("robot: ");
		rkllm_run(llmHandle, &rkllm_input, &rkllm_infer_params, NULL);
	}

	rkllm_destroy(llmHandle);

	return 0;
}