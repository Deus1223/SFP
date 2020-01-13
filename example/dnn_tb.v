`timescale 1ns/1ps
module rtl_dnn_tb;

wire cmd_start;
wire [1:0] cmd;
wire cmd_buf_sel;
wire [2:0] cmd_cur_layer;
wire cmd_done;

wire dnn_done;

wire [1:0] mem_sel;
wire [1:0] mem_we;
wire [11:0] mem_addr;
wire [31:0] mem_din;
wire [31:0] mem_dout;

reg s_ext_valid;
reg [31:0] s_ext_addr;
reg  s_ext_we;
reg [31:0] s_ext_din;
wire [31:0] s_ext_dout;
wire s_ext_rdy;

wire m_ext_valid;
wire [31:0] m_ext_addr;
wire m_ext_we;
wire [31:0] m_ext_din;
wire [31:0] m_ext_dout;
wire m_ext_rdy;

wire [2:0] hidden_layers;
wire [8:0] input_neurons;
wire [8:0] hl0_neurons;
wire [8:0] hl0_stride;
wire [8:0] hl1_neurons;
wire [8:0] hl1_stride;
wire [8:0] hl2_neurons;
wire [8:0] hl2_stride;
wire [8:0] hl3_neurons;
wire [8:0] hl3_stride;
wire [8:0] ol_neurons;
wire [8:0] ol_stride;
wire [8:0] ol_input_neurons;

wire dnn_start, dnn_core_rst_n;

integer i, j, cmd_cnt;

reg clk, rst_n;

initial clk = 1'b0;
always #5 clk = ~clk;

initial begin
	rst_n = 1'b1;
	#22;
	rst_n = 1'b0;
	#20;
	rst_n = 1'b1;
end

initial begin
	$dumpfile("./dnn_dump.vcd");
	$dumpvars(0);
`ifdef GATE_SIM
	//$sdf_annotate("./syn/ccu_biu.sdf", dut_0);

`endif
end

initial begin
	#100000000;
	$finish;
end

`ifdef GATE_SIM
ccu_biu dut_0(
`else
biu dut_0(
`endif
	.clk(clk),
	.rst_n(rst_n),

	// Slave bus ports
	.s_ext_valid(s_ext_valid),
	.s_ext_addr(s_ext_addr[15:0]),
	.s_ext_we(s_ext_we),
	.s_ext_din(s_ext_din),
	.s_ext_dout(s_ext_dout),
	.s_ext_rdy(s_ext_rdy),

	// Master bus ports
	.m_ext_valid(m_ext_valid),
	.m_ext_addr(m_ext_addr),
	.m_ext_we(m_ext_we),
	.m_ext_din(m_ext_din),
	.m_ext_dout(m_ext_dout),
	.m_ext_rdy(m_ext_rdy),

	// interrupt out
	.int_o(),
	
	// cfg
	.hidden_layers(hidden_layers),
	.input_neurons(input_neurons),
	.hl0_neurons(hl0_neurons),
	.hl0_stride(hl0_stride),
	.hl1_neurons(hl1_neurons),
	.hl1_stride(hl1_stride),
	.hl2_neurons(hl2_neurons),
	.hl2_stride(hl2_stride),
	.hl3_neurons(hl3_neurons),
	.hl3_stride(hl3_stride),
	.ol_neurons(ol_neurons),
	.ol_stride(ol_stride),
	.ol_input_neurons(ol_input_neurons),

	// cmd
	.cmd_start(cmd_start), 
	.cmd(cmd), 
	.cmd_buf_sel(cmd_buf_sel),
	.cmd_cur_layer(cmd_cur_layer),
	.cmd_done(cmd_done),

	.dnn_core_rst_n(dnn_core_rst_n), 
	.dnn_start(dnn_start), 
	.dnn_done(dnn_done),

	// mem
	.mem_sel(mem_sel),
	.mem_addr(mem_addr),
	.mem_din(mem_din),
	.mem_dout(mem_dout),
	.mem_we(mem_we)
);

`ifdef GATE_SIM
ccu_dnn_acc dnn_0(
`else
dnn_acc dnn_0(
`endif
	.clk(clk),
	.rst_n(dnn_core_rst_n),

	.hidden_layers(hidden_layers),
	.input_neurons(input_neurons),
	.hl0_neurons(hl0_neurons),
	.hl0_stride(hl0_stride),
	.hl1_neurons(hl1_neurons),
	.hl1_stride(hl1_stride),
	.hl2_neurons(hl2_neurons),
	.hl2_stride(hl2_stride),
	.hl3_neurons(hl3_neurons),
	.hl3_stride(hl3_stride),
	.ol_neurons(ol_neurons),
	.ol_stride(ol_stride),
	.ol_input_neurons(ol_input_neurons),

	.mem_addr(mem_addr),
	.mem_sel(mem_sel),
	.mem_we(mem_we),
	.mem_din(mem_din),
	.mem_dout(mem_dout),

	.dnn_start(dnn_start),
	.dnn_done(dnn_done),

	.cmd_start(cmd_start),
	.cmd(cmd),
	.cmd_buf_sel(cmd_buf_sel),
	.cmd_cur_layer(cmd_cur_layer),
	.cmd_done(cmd_done)
);

bus_behav #(
	.dly(0)
)
bus_0(
	.clk(clk),
	.valid(m_ext_valid),
	.addr(m_ext_addr),
	.we(m_ext_we),
	.din(m_ext_din),
	.dout(m_ext_dout),
	.rdy(m_ext_rdy)
);

// 32'h3f80_0000 is 1.0 (fp32)

localparam cfg_param_ptr = 32'h1f00_0000;
localparam cfg_optr_base = 32'h1ff0_0000;
//localparam cfg_iptr_base = 32'h1e00_0000;
//localparam cfg_iptr_base = 32'h1e00_0600;
//localparam cfg_iptr_base = 32'h1e00_0C00;
localparam cfg_iptr_base = 32'h1e00_0000;

//localparam cfg_hidden_layers = 9'h3;
//localparam cfg_hidden_layers = 9'h3;
//localparam cfg_input_neurons = 9'hff;
//localparam cfg_input_neurons = 9'h81;

//localparam cfg_hl0_neurons = 9'h0f;	//16
//localparam cfg_hl0_neurons = 9'hff;	//256
//localparam cfg_hl0_neurons = 9'h1ff;	//512
localparam cfg_hl0_stride = 9'h0f;	//16 256
//localparam cfg_hl0_stride = 9'h07;	//512

//localparam cfg_hl1_neurons = 9'hf;	//16
//localparam cfg_hl1_neurons = 9'hff;	//256
//localparam cfg_hl1_neurons = 9'h1ff;	//512
localparam cfg_hl1_stride = 9'hf;	//16 256
//localparam cfg_hl1_stride = 9'h07;  //512

//localparam cfg_hl2_neurons = 9'hf;	//16
//localparam cfg_hl2_neurons = 9'hff;	//256
//localparam cfg_hl2_neurons = 9'h1ff;	//512
localparam cfg_hl2_stride = 9'hf;	//16 256
//localparam cfg_hl2_stride = 9'h07;  //512

//localparam cfg_ol_neurons = 9'h09;
//localparam cfg_ol_neurons = 9'h81; //vc
localparam cfg_ol_stride = 9'h09;
//localparam cfg_ol_stride = 9'h7; //512
//localparam cfg_ol_neurons = 9'h0f;

/* //512
localparam cfg_hl0_neurons = 9'h1ff;	//512
localparam cfg_hl0_stride = 9'h07;	//512
localparam cfg_hl1_neurons = 9'h1ff;	//512
localparam cfg_hl1_stride = 9'h07;	//512
localparam cfg_hl2_neurons = 9'h1ff;	//512
localparam cfg_hl2_stride = 9'h07;	//512
localparam cfg_ol_neurons = 9'h09;
localparam cfg_ol_stride = 9'h09;
*/

/*
// 3x8x7x4
localparam cfg_hidden_layers = 9'h2;
localparam cfg_input_neurons = 9'h2;

localparam cfg_hl0_neurons = 9'h7;
localparam cfg_hl0_stride = 9'h2;

localparam cfg_hl1_neurons = 9'h6;
localparam cfg_hl1_stride = 9'h3;

localparam cfg_ol_neurons = 9'h3;
localparam cfg_ol_stride = 9'h1;
*/


/*
// 8x8x8x8
localparam cfg_hidden_layers = 9'h2;
localparam cfg_input_neurons = 9'h7;

localparam cfg_hl0_neurons = 9'h7;
localparam cfg_hl0_stride = 9'h2;

localparam cfg_hl1_neurons = 9'h7;
localparam cfg_hl1_stride = 9'h2;

localparam cfg_ol_neurons = 9'h7;
localparam cfg_ol_stride = 9'h2;
*/

reg[8:0] cfg_hidden_layers = 9'h3;
reg[8:0] cfg_input_neurons = 9'hff;
reg[8:0] cfg_hl0_neurons = 9'h0f;
reg[8:0] cfg_hl1_neurons = 9'h0f;
reg[8:0] cfg_hl2_neurons = 9'h0f;
reg[8:0] cfg_ol_neurons = 9'h09;
integer fp_config;
initial begin
	fp_config = $fopen("./data/config.txt", "r");
	$fscanf(fp_config, "%d\n", cfg_hidden_layers);
	$fscanf(fp_config, "%d\n", cfg_input_neurons);
	$fscanf(fp_config, "%d\n", cfg_hl0_neurons);
	$fscanf(fp_config, "%d\n", cfg_hl1_neurons);
	$fscanf(fp_config, "%d\n", cfg_hl2_neurons);
	$fscanf(fp_config, "%d\n", cfg_ol_neurons);
	$fclose(fp_config);

	s_ext_valid = 1'b0;
	s_ext_addr = 32'h0;
	s_ext_din = 32'h0;
	s_ext_we = 1'b0;

	@(posedge rst_n);

	/*
	dut_0.cfg_param_ptr = 32'h1f00_0000;
	dut_0.cfg_optr = 32'h1ff0_0000;
	dut_0.cfg_iptr = 32'h1e00_0000;
	dut_0.cfg_hidden_layers = 9'h1;
	dut_0.cfg_input_neurons = 9'hff;
	//dut_0.cfg_input_neurons = 9'h04;
	dut_0.cfg_hl0_neurons = 9'h0f;
	dut_0.cfg_hl0_stride = 9'h0f;
	dut_0.cfg_ol_neurons = 9'h09;
	dut_0.cfg_ol_stride = 9'h09;
	*/
	@(negedge clk);
	s_ext_valid = 1'b1;
	s_ext_addr = 32'h8000_2000;
	s_ext_din = {7'b0, cfg_hidden_layers, 7'b0, cfg_input_neurons};
	s_ext_we = 1'b1;
	@(negedge clk);
	s_ext_valid = 1'b0;
	s_ext_we = 1'b0;

	@(negedge clk);
	s_ext_valid = 1'b1;
	s_ext_addr = 32'h8000_2004;
	s_ext_din = {7'b0, cfg_hl0_neurons, 7'b0, cfg_hl0_stride};
	s_ext_we = 1'b1;
	@(negedge clk);
	s_ext_valid = 1'b0;
	s_ext_we = 1'b0;

	@(negedge clk);
	s_ext_valid = 1'b1;
	s_ext_addr = 32'h8000_2008;
	s_ext_din = {7'b0, cfg_hl1_neurons, 7'b0, cfg_hl1_stride};
	s_ext_we = 1'b1;
	@(negedge clk);
	s_ext_valid = 1'b0;
	s_ext_we = 1'b0;

	@(negedge clk);
	s_ext_valid = 1'b1;
	s_ext_addr = 32'h8000_200C;
	s_ext_din = {7'b0, cfg_hl2_neurons, 7'b0, cfg_hl2_stride};
	s_ext_we = 1'b1;
	@(negedge clk);
	s_ext_valid = 1'b0;
	s_ext_we = 1'b0;

	@(negedge clk);
	s_ext_valid = 1'b1;
	s_ext_addr = 32'h8000_2014;
	s_ext_din = {7'b0, cfg_ol_neurons, 7'b0, cfg_ol_stride};
	s_ext_we = 1'b1;
	@(negedge clk);
	s_ext_valid = 1'b0;
	s_ext_we = 1'b0;

	@(negedge clk);
	s_ext_valid = 1'b1;
	s_ext_addr = 32'h8000_2018;
	s_ext_din = cfg_iptr_base;
	s_ext_we = 1'b1;
	@(negedge clk);
	s_ext_valid = 1'b0;
	s_ext_we = 1'b0;

	@(negedge clk);
	s_ext_valid = 1'b1;
	s_ext_addr = 32'h8000_201C;
	s_ext_din = cfg_optr_base;
	s_ext_we = 1'b1;
	@(negedge clk);
	s_ext_valid = 1'b0;
	s_ext_we = 1'b0;

	@(negedge clk);
	s_ext_valid = 1'b1;
	s_ext_addr = 32'h8000_2020;
	s_ext_din = cfg_param_ptr;
	s_ext_we = 1'b1;
	@(negedge clk);
	s_ext_valid = 1'b0;
	s_ext_we = 1'b0;

	#100;
	@(negedge clk);
	s_ext_valid = 1'b1;
	s_ext_addr = 32'h8000_4000;
	s_ext_din = {1'b1, 31'h1};
	s_ext_we = 1'b1;
	@(negedge clk);
	s_ext_valid = 1'b0;
	s_ext_we = 1'b0;


	#10000;
	@(negedge clk);
	s_ext_valid = 1'b1;
	s_ext_addr = 32'h8000_4000;
	s_ext_din = {1'b1, 1'b1, 30'h1};
	s_ext_we = 1'b1;
	@(negedge clk);
	s_ext_valid = 1'b0;
	s_ext_we = 1'b0;


	$display("Image#00000(hl):");
	for (i=0; i<16; i=i+1) begin
		//$display("  FP32 %1d %08h", i, dbuf[1][i]);
		//$display("  FP32 %1d %08h", i, dbuf_1.mem[i]);
	end

	$display("Image#00000(ol):");
	for (i=0; i<10; i=i+1) begin
		//$display("  FP32 %1d %08h", i, dbuf[0][i]);
		//$display("  FP32 %1d %08h", i, dbuf_0.mem[i]);
	end


	#20;
	@(dnn_done==1);

	/*#1000;
	@(negedge clk);
	s_ext_valid = 1'b1;
	s_ext_addr = 32'h8000_2018;
	s_ext_din = cfg_iptr_base + 32'd256*32'd2;
	s_ext_we = 1'b1;
	@(negedge clk);
	s_ext_valid = 1'b0;
	s_ext_we = 1'b0;

	@(negedge clk);
	s_ext_valid = 1'b1;
	s_ext_addr = 32'h8000_201C;
	//s_ext_din = cfg_optr_base + 32'd16*32'd4;
	s_ext_din = cfg_optr_base + 32'd5*32'd4;
	s_ext_we = 1'b1;
	@(negedge clk);
	s_ext_valid = 1'b0;
	s_ext_we = 1'b0;
	@(negedge clk);
	s_ext_valid = 1'b1;
	s_ext_addr = 32'h8000_4000;
	s_ext_din = {1'b1, 1'b1, 14'h0, 1'b0, 1'b0, 1'b1, 13'b0};
	s_ext_we = 1'b1;
	@(negedge clk);
	s_ext_valid = 1'b0;
	s_ext_we = 1'b0;

	$display("Image#00001(hl):");
	for (i=0; i<16; i=i+1) begin
		//$display("  FP32 %1d %08h", i, dbuf[1][i]);
		//$display("  FP32 %1d %08h", i, dbuf_1.mem[i]);
	end

	$display("Image#00001(ol):");
	for (i=0; i<10; i=i+1) begin
		//$display("  FP32 %1d %08h", i, dbuf[0][i]);
		//$display("  FP32 %1d %08h", i, dbuf_0.mem[i]);
	end

	#20;
	@(dnn_done==1);

	#1000;
	@(negedge clk);
	s_ext_valid = 1'b1;
	s_ext_addr = 32'h8000_2018;
	s_ext_din = cfg_iptr_base + 32'd256*32'd2*2;
	s_ext_we = 1'b1;
	@(negedge clk);
	s_ext_valid = 1'b0;
	s_ext_we = 1'b0;

	@(negedge clk);
	s_ext_valid = 1'b1;
	s_ext_addr = 32'h8000_201C;
	//s_ext_din = cfg_optr_base + 32'd16*32'd4;
	s_ext_din = cfg_optr_base + 32'd5*32'd4*2;
	s_ext_we = 1'b1;
	@(negedge clk);
	s_ext_valid = 1'b0;
	s_ext_we = 1'b0;
	@(negedge clk);
	s_ext_valid = 1'b1;
	s_ext_addr = 32'h8000_4000;
	s_ext_din = {1'b1, 1'b1, 30'h1};
	s_ext_we = 1'b1;
	@(negedge clk);
	s_ext_valid = 1'b0;
	s_ext_we = 1'b0;

	$display("Image#00002(hl):");
	for (i=0; i<16; i=i+1) begin
		//$display("  FP32 %1d %08h", i, dbuf[1][i]);
		//$display("  FP32 %1d %08h", i, dbuf_1.mem[i]);
	end

	$display("Image#00002(ol):");
	for (i=0; i<10; i=i+1) begin
		//$display("  FP32 %1d %08h", i, dbuf[0][i]);
		//$display("  FP32 %1d %08h", i, dbuf_0.mem[i]);
	end

	#20;
	@(dnn_done==1);


	@(negedge clk);
	s_ext_valid = 1'b1;
	s_ext_addr = 32'h8000_0000;
	@(negedge clk);
	s_ext_valid = 1'b0;
	#100;

	@(negedge clk);
	s_ext_valid = 1'b1;
	s_ext_addr = 32'h8000_1000;
	@(negedge clk);
	s_ext_valid = 1'b0;
	#100;

	@(negedge clk);
	s_ext_valid = 1'b1;
	s_ext_addr = 32'h8000_8000;
	@(negedge clk);
	s_ext_valid = 1'b0;
	#100;

	@(negedge clk);
	s_ext_valid = 1'b1;
	s_ext_addr = 32'h8000_C000;
	@(negedge clk);
	s_ext_valid = 1'b0;*/
	#100;

	for (i=0; i<1; i=i+1) begin
		$display("Image #%1d:", i);
		//for (j=0; j<16; j=j+1)
			//$display("  WORD32 %1d %08h", j, bus_0.mem[32505856/4 + j + i*16]);
		for (j=0; j<5; j=j+1)
			$display("  WORD32 %1d %08h", j, bus_0.mem[32505856/4 + j + i*5]);
	end

	#1000;
	$finish;
end

initial cmd_cnt = 0;
always@(posedge cmd_start) begin
	cmd_cnt = cmd_cnt + 1;
	$display("cnt=%d", cmd_cnt);
end

endmodule
