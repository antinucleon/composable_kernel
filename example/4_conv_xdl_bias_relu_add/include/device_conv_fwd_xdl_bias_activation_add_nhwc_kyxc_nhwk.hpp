#ifndef DEVICE_CONV_FWD_XDL_BIAS_ACTIVATION_ADD_NHWC_KYXC_NHWK_HPP
#define DEVICE_CONV_FWD_XDL_BIAS_ACTIVATION_ADD_NHWC_KYXC_NHWK_HPP

#include <iostream>
#include "device.hpp"
#include "device_base.hpp"
#include "device_conv.hpp"
#include "common_header.hpp"
#include "tensor_layout.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm_xdlops_v2r5.hpp"
#include "example/4_conv_xdl_bias_relu_add/include/device_conv_fwd_xdl_bias_activation_add.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// specialization for 2D conv: in[n, hi, wi, c] * wei[k, y, x, c] = out[n, ho, wo, k]
template <typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename AccDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t K0PerBlock,
          ck::index_t K1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
          typename ABlockTransferThreadSliceLengths_K0_M_K1,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_K1,
          typename BBlockTransferThreadSliceLengths_K0_N_K1,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_K1,
          ck::index_t CThreadTransferSrcDstVectorDim,
          ck::index_t CThreadTransferDstScalarPerVector,
          bool ABlockLdsAddExtraM,
          bool BBlockLdsAddExtraN>
struct DeviceConvFwdXdl_bias_activation_add<
    2,                                        // ck::index_t NDimSpatial,
    InDataType,                               // typename InDataType,
    WeiDataType,                              // typename WeiDataType,
    OutDataType,                              // typename OutDataType,
    AccDataType,                              // typename AccDataType,
    ck::tensor_layout::convolution::NHWC,     // typename InLayout,
    ck::tensor_layout::convolution::KYXC,     // typename WeiLayout,
    ck::tensor_layout::convolution::NHWK,     // typename OutLayout,
    InElementwiseOperation,                   // typename InElementwiseOperation,
    WeiElementwiseOperation,                  // typename WeiElementwiseOperation,
    OutElementwiseOperation,                  // typename OutElementwiseOperation,
    BlockSize,                                // ck::index_t BlockSize,
    MPerBlock,                                // ck::index_t MPerBlock,
    NPerBlock,                                // ck::index_t NPerBlock,
    K0PerBlock,                               // ck::index_t K0PerBlock,
    K1,                                       // ck::index_t K1,
    MPerXDL,                                  // ck::index_t MPerXDL,
    NPerXDL,                                  // ck::index_t NPerXDL,
    MXdlPerWave,                              // ck::index_t MXdlPerWave,
    NXdlPerWave,                              // ck::index_t NXdlPerWave,
    ABlockTransferThreadSliceLengths_K0_M_K1, // typename ABlockTransferThreadSliceLengths_K0_M_K1,
    ABlockTransferThreadClusterLengths_K0_M_K1, // typename
                                                // ABlockTransferThreadClusterLengths_K0_M_K1,
    ABlockTransferThreadClusterArrangeOrder,    // typename ABlockTransferThreadClusterArrangeOrder,
    ABlockTransferSrcAccessOrder,               // typename ABlockTransferSrcAccessOrder,
    ABlockTransferSrcVectorDim,                 // ck::index_t ABlockTransferSrcVectorDim,
    ABlockTransferSrcScalarPerVector,           // ck::index_t ABlockTransferSrcScalarPerVector,
    ABlockTransferDstScalarPerVector_K1,        // ck::index_t ABlockTransferDstScalarPerVector_K1,
    BBlockTransferThreadSliceLengths_K0_N_K1, // typename BBlockTransferThreadSliceLengths_K0_N_K1,
    BBlockTransferThreadClusterLengths_K0_N_K1, // typename
                                                // BBlockTransferThreadClusterLengths_K0_N_K1,
    BBlockTransferThreadClusterArrangeOrder,    // typename BBlockTransferThreadClusterArrangeOrder,
    BBlockTransferSrcAccessOrder,               // typename BBlockTransferSrcAccessOrder,
    BBlockTransferSrcVectorDim,                 // ck::index_t BBlockTransferSrcVectorDim,
    BBlockTransferSrcScalarPerVector,           // ck::index_t BBlockTransferSrcScalarPerVector,
    BBlockTransferDstScalarPerVector_K1,        // ck::index_t BBlockTransferDstScalarPerVector_K1,
    CThreadTransferSrcDstVectorDim,             // ck::index_t CThreadTransferSrcDstVectorDim,
    CThreadTransferDstScalarPerVector,          // ck::index_t CThreadTransferDstScalarPerVector,
    ABlockLdsAddExtraM,                         // bool ABlockLdsAddExtraM,
    BBlockLdsAddExtraN                          // bool BBlockLdsAddExtraN>
    > : public BaseOperator
{
    using ADataType = InDataType;
    using BDataType = WeiDataType;
    using CDataType = OutDataType;

    // TODO make A/B datatype different
    using ABDataType = InDataType;

    // TODO make it support any # of spatial dimensions
    static constexpr index_t NDimSpatial = 2;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};

    static constexpr auto K1Number     = Number<K1>{};
    static constexpr auto GemmK1Number = K1Number;

    static auto
    MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(ck::index_t N,
                                                    ck::index_t K,
                                                    ck::index_t C,
                                                    std::vector<ck::index_t> input_spatial_lengths,
                                                    std::vector<ck::index_t> filter_spatial_lengths,
                                                    std::vector<ck::index_t> output_spatial_lengths,
                                                    std::vector<ck::index_t> conv_filter_strides,
                                                    std::vector<ck::index_t> conv_filter_dilations,
                                                    std::vector<ck::index_t> input_left_pads,
                                                    std::vector<ck::index_t> input_right_pads)
    {
        using namespace ck;

        const index_t Hi = input_spatial_lengths[0];
        const index_t Wi = input_spatial_lengths[1];

        const index_t Ho = output_spatial_lengths[0];
        const index_t Wo = output_spatial_lengths[1];

        const index_t Y = filter_spatial_lengths[0];
        const index_t X = filter_spatial_lengths[1];

        const index_t ConvStrideH = conv_filter_strides[0];
        const index_t ConvStrideW = conv_filter_strides[1];

        const index_t ConvDilationH = conv_filter_dilations[0];
        const index_t ConvDilationW = conv_filter_dilations[1];

        const index_t InLeftPadH = input_left_pads[0];
        const index_t InLeftPadW = input_left_pads[1];

        const index_t InRightPadH = input_right_pads[0];
        const index_t InRightPadW = input_right_pads[1];

        const index_t GemmMRaw = N * Ho * Wo;
        const index_t GemmN    = K;
        const index_t GemmK    = Y * X * C;

        const auto GemmMPad = math::integer_least_multiple(GemmMRaw, MPerBlock) - GemmMRaw;

        const auto GemmM = GemmMRaw + GemmMPad;

        assert(GemmK % GemmK1Number == 0);

        const index_t GemmK0 = GemmK / GemmK1Number;

        // A: input tensor
        const auto in_n_hi_wi_c_grid_desc =
            make_naive_tensor_descriptor_packed(make_tuple(N, Hi, Wi, C));

        const auto in_n_hip_wip_c_grid_desc = transform_tensor_descriptor(
            in_n_hi_wi_c_grid_desc,
            make_tuple(make_pass_through_transform(N),
                       make_pad_transform(Hi, InLeftPadH, InRightPadH),
                       make_pad_transform(Wi, InLeftPadW, InRightPadW),
                       make_pass_through_transform(C)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        const auto in_n_y_ho_x_wo_c_grid_desc = transform_tensor_descriptor(
            in_n_hip_wip_c_grid_desc,
            make_tuple(
                make_pass_through_transform(N),
                make_embed_transform(make_tuple(Y, Ho), make_tuple(ConvDilationH, ConvStrideH)),
                make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW)),
                make_pass_through_transform(C)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

        const auto in_gemmk_gemmmraw_grid_desc =
            transform_tensor_descriptor(in_n_y_ho_x_wo_c_grid_desc,
                                        make_tuple(make_merge_transform(make_tuple(Y, X, C)),
                                                   make_merge_transform(make_tuple(N, Ho, Wo))),
                                        make_tuple(Sequence<1, 3, 5>{}, Sequence<0, 2, 4>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));

        const auto in_gemmk0_gemmmraw_gemmk1_grid_desc = transform_tensor_descriptor(
            in_gemmk_gemmmraw_grid_desc,
            make_tuple(make_unmerge_transform(make_tuple(GemmK0, GemmK1Number)),
                       make_pass_through_transform(GemmMRaw)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

        const auto in_gemmk0_gemmm_gemmk1_grid_desc =
            transform_tensor_descriptor(in_gemmk0_gemmmraw_gemmk1_grid_desc,
                                        make_tuple(make_pass_through_transform(GemmK0),
                                                   make_right_pad_transform(GemmMRaw, GemmMPad),
                                                   make_pass_through_transform(GemmK1Number)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        // B: weight tensor
        const auto wei_k_yxc_grid_desc =
            make_naive_tensor_descriptor_packed(make_tuple(K, Y * X * C));

        const auto wei_gemmk_gemmn_grid_desc = transform_tensor_descriptor(
            wei_k_yxc_grid_desc,
            make_tuple(make_pass_through_transform(K), make_pass_through_transform(Y * X * C)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<1>{}, Sequence<0>{}));

        const auto wei_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
            wei_gemmk_gemmn_grid_desc,
            make_tuple(make_unmerge_transform(make_tuple(GemmK0, GemmK1Number)),
                       make_pass_through_transform(GemmN)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

        // C: output tensor
        const auto out_nhowo_k_grid_desc =
            make_naive_tensor_descriptor_packed(make_tuple(N * Ho * Wo, K));

        const auto out_gemmmraw_gemmn_grid_desc = transform_tensor_descriptor(
            out_nhowo_k_grid_desc,
            make_tuple(make_pass_through_transform(N * Ho * Wo), make_pass_through_transform(K)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        const auto out_gemmm_gemmn_grid_desc =
            transform_tensor_descriptor(out_gemmmraw_gemmn_grid_desc,
                                        make_tuple(make_right_pad_transform(GemmMRaw, GemmMPad),
                                                   make_pass_through_transform(GemmN)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));

        // C0: bias tensor: assume a contiguous vector
        const auto bias_grid_desc_gemmm_gemmn =
            make_naive_tensor_descriptor(make_tuple(GemmM, GemmN), make_tuple(0, 1));

        // C1: residual tensor: assume same layout as output tensor
        const auto resi_grid_desc_gemmm_gemmn = out_gemmm_gemmn_grid_desc;

        return make_tuple(in_gemmk0_gemmm_gemmk1_grid_desc,
                          wei_gemmk0_gemmn_gemmk1_grid_desc,
                          out_gemmm_gemmn_grid_desc,
                          bias_grid_desc_gemmm_gemmn,
                          resi_grid_desc_gemmm_gemmn);
    }

    using ABCGridDescs = decltype(MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(
        1, 1, 1, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}));

    using AGridDesc_K0_M_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I0])>;
    using BGridDesc_K0_N_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I1])>;
    using CGridDesc_M_N     = remove_cvref_t<decltype(ABCGridDescs{}[I2])>;
    using C0GridDesc_M_N    = remove_cvref_t<decltype(ABCGridDescs{}[I3])>;
    using C1GridDesc_M_N    = remove_cvref_t<decltype(ABCGridDescs{}[I4])>;

    // TODO remove these hacks
    static constexpr auto a_k0_m_k1_grid_step_hacks = make_tuple(
        make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0>{},   // 0+: K0
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0>{},   // 1+: M
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0>{}),  // 2+: K1
        make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0>{},   // 0-: K0
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0>{},   // 1-: M
                   Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0>{})); // 2-: K1

    static constexpr auto b_k0_n_k1_grid_step_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},   // 0+: K0
                              Sequence<0, 0, 0, 0, 0>{},   // 1+: N
                              Sequence<0, 0, 0, 0, 0>{}),  // 2+: K1
                   make_tuple(Sequence<0, 0, 0, 0, 0>{},   // 0-: K0
                              Sequence<0, 0, 0, 0, 0>{},   // 1-: N
                              Sequence<0, 0, 0, 0, 0>{})); // 2-: K1

    static constexpr auto c_m0_n0_m1_n1_m2_m3_m4_n2_grid_step_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 0+: M0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 1+: N0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 2+: M1
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 3+: N1
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 4+: M2
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 5+: M3
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 6+: M4
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}),  // 7+: N2
                   make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 0-: M0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 1-: N0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 2-: M1
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 3-: N1
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 4-: M2
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 5-: M3
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 6-: M4
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{})); // 7-: N2

    static constexpr auto a_k0_m_k1_grid_move_slice_window_step_hacks =
        Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0>{};

    static constexpr auto b_k0_n_k1_grid_move_slice_window_step_hacks = Sequence<0, 0, 0, 0, 0>{};

    // GridwiseGemm
    using GridwiseGemm = GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r5<
        BlockSize,
        ABDataType, // TODO: distinguish A/B datatype
        AccDataType,
        CDataType,
        InMemoryDataOperationEnum_t::Set,
        AGridDesc_K0_M_K1,
        BGridDesc_K0_N_K1,
        CGridDesc_M_N,
        C0GridDesc_M_N,
        C1GridDesc_M_N,
        InElementwiseOperation,
        WeiElementwiseOperation,
        OutElementwiseOperation,
        MPerBlock,
        NPerBlock,
        K0PerBlock,
        MPerXDL,
        NPerXDL,
        K1,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadSliceLengths_K0_M_K1,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        Sequence<1, 0, 2>, // ABlockTransferThreadClusterArrangeOrder,
        Sequence<1, 0, 2>, // ABlockTransferSrcAccessOrder,
        2,                 // ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        false, // AThreadTransferSrcResetCoordinateAfterRun,
        BBlockTransferThreadSliceLengths_K0_N_K1,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        Sequence<1, 0, 2>, // BBlockTransferThreadClusterArrangeOrder,
        Sequence<1, 0, 2>, // BBlockTransferSrcAccessOrder,
        2,                 // BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false,                            // BThreadTransferSrcResetCoordinateAfterRun,
        Sequence<2, 3, 0, 1, 7, 5, 4, 6>, // CThreadTransferSrcDstAccessOrder,
        7,                                // CThreadTransferSrcDstVectorDim,
        CThreadTransferDstScalarPerVector,
        decltype(a_k0_m_k1_grid_step_hacks),                   //  AGridStepHacks,
        decltype(b_k0_n_k1_grid_step_hacks),                   //  BGridStepHacks,
        decltype(c_m0_n0_m1_n1_m2_m3_m4_n2_grid_step_hacks),   //  CGridStepHacks,
        decltype(a_k0_m_k1_grid_move_slice_window_step_hacks), //  AGridMoveSliceWindowStepHacks,
        decltype(b_k0_n_k1_grid_move_slice_window_step_hacks), //  BGridMoveSliceWindowStepHacks,
        false,                                                 // CAccessOrderMRepeatNRepeat,
        ABlockLdsAddExtraM,
        BBlockLdsAddExtraN>;

    using CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2 =
        decltype(GridwiseGemm::MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(CGridDesc_M_N{}));

    using C0GridDesc_M0_N0_M1_N1_M2_M3_M4_N2 =
        decltype(GridwiseGemm::MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(C0GridDesc_M_N{}));

    using C1GridDesc_M0_N0_M1_N1_M2_M3_M4_N2 =
        decltype(GridwiseGemm::MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(C1GridDesc_M_N{}));

    using Block2CTileMap = decltype(GridwiseGemm::MakeBlock2CTileMap(CGridDesc_M_N{}, 1, 1));

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const InDataType* p_in_grid,
                 const WeiDataType* p_wei_grid,
                 OutDataType* p_out_grid,
                 const OutDataType* p_bias_grid,
                 const OutDataType* p_resi_grid,
                 ck::index_t N,
                 ck::index_t K,
                 ck::index_t C,
                 std::vector<ck::index_t> input_spatial_lengths,
                 std::vector<ck::index_t> filter_spatial_lengths,
                 std::vector<ck::index_t> output_spatial_lengths,
                 std::vector<ck::index_t> conv_filter_strides,
                 std::vector<ck::index_t> conv_filter_dilations,
                 std::vector<ck::index_t> input_left_pads,
                 std::vector<ck::index_t> input_right_pads,
                 ck::index_t M01,
                 ck::index_t N01,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op)
            : p_a_grid_{p_in_grid},
              p_b_grid_{p_wei_grid},
              p_c_grid_{p_out_grid},
              p_c0_grid_{p_bias_grid},
              p_c1_grid_{p_resi_grid},
              a_grid_desc_k0_m_k1_{},
              b_grid_desc_k0_n_k1_{},
              c_grid_desc_m_n_{},
              c0_grid_desc_m_n_{},
              c1_grid_desc_m_n_{},
              c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_{},
              c0_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_{},
              c1_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_{},
              block_2_ctile_map_{},
              M01_{M01},
              N01_{N01},
              in_element_op_{in_element_op},
              wei_element_op_{wei_element_op},
              out_element_op_{out_element_op}
        {
            const auto descs = DeviceConvFwdXdl_bias_activation_add::
                MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(N,
                                                                K,
                                                                C,
                                                                input_spatial_lengths,
                                                                filter_spatial_lengths,
                                                                output_spatial_lengths,
                                                                conv_filter_strides,
                                                                conv_filter_dilations,
                                                                input_left_pads,
                                                                input_right_pads);

            a_grid_desc_k0_m_k1_ = descs[I0];
            b_grid_desc_k0_n_k1_ = descs[I1];
            c_grid_desc_m_n_     = descs[I2];
            c0_grid_desc_m_n_    = descs[I3];
            c1_grid_desc_m_n_    = descs[I4];

            if(GridwiseGemm::CheckValidity(
                   a_grid_desc_k0_m_k1_, b_grid_desc_k0_n_k1_, c_grid_desc_m_n_, M01_, N01_))
            {
                c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_ =
                    GridwiseGemm::MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(c_grid_desc_m_n_);

                c0_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_ =
                    GridwiseGemm::MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(c0_grid_desc_m_n_);

                c1_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_ =
                    GridwiseGemm::MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(c1_grid_desc_m_n_);

                block_2_ctile_map_ = GridwiseGemm::MakeBlock2CTileMap(c_grid_desc_m_n_, M01, N01);
            }
        }

        //  private:
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        CDataType* p_c_grid_;
        const CDataType* p_c0_grid_;
        const CDataType* p_c1_grid_;
        AGridDesc_K0_M_K1 a_grid_desc_k0_m_k1_;
        BGridDesc_K0_N_K1 b_grid_desc_k0_n_k1_;
        CGridDesc_M_N c_grid_desc_m_n_;
        C0GridDesc_M_N c0_grid_desc_m_n_;
        C1GridDesc_M_N c1_grid_desc_m_n_;
        CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2 c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_;
        C0GridDesc_M0_N0_M1_N1_M2_M3_M4_N2 c0_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_;
        C1GridDesc_M0_N0_M1_N1_M2_M3_M4_N2 c1_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_;
        Block2CTileMap block_2_ctile_map_;
        index_t M01_;
        index_t N01_;
        InElementwiseOperation in_element_op_;
        WeiElementwiseOperation wei_element_op_;
        OutElementwiseOperation out_element_op_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceConvFwdXdl_bias_activation_add::Argument;

        float Run(const Argument& arg, int nrepeat = 1)
        {
            {
                std::cout << "arg.a_grid_desc_k0_m_k1_{" << arg.a_grid_desc_k0_m_k1_.GetLength(I0)
                          << ", " << arg.a_grid_desc_k0_m_k1_.GetLength(I1) << ", "
                          << arg.a_grid_desc_k0_m_k1_.GetLength(I2) << "}" << std::endl;

                std::cout << "arg.b_grid_desc_k0_n_k1_{" << arg.b_grid_desc_k0_n_k1_.GetLength(I0)
                          << ", " << arg.b_grid_desc_k0_n_k1_.GetLength(I1) << ", "
                          << arg.b_grid_desc_k0_n_k1_.GetLength(I2) << "}" << std::endl;

                std::cout << "arg.c_grid_desc_m_n_{ " << arg.c_grid_desc_m_n_.GetLength(I0) << ", "
                          << arg.c_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;

                std::cout << "arg.c0_grid_desc_m_n_{ " << arg.c0_grid_desc_m_n_.GetLength(I0)
                          << ", " << arg.c0_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;

                std::cout << "arg.c1_grid_desc_m_n_{ " << arg.c1_grid_desc_m_n_.GetLength(I0)
                          << ", " << arg.c1_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;
            }

            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_k0_m_k1_,
                                            arg.b_grid_desc_k0_n_k1_,
                                            arg.c_grid_desc_m_n_,
                                            arg.M01_,
                                            arg.N01_))
            {
                throw std::runtime_error(
                    "wrong! GridwiseGemm_km_kn_m0m1n0n1_xdlops_v2r5 has invalid setting");
            }

            const index_t grid_size = GridwiseGemm::CalculateGridSize(arg.c_grid_desc_m_n_);

            const auto K0 = arg.a_grid_desc_k0_m_k1_.GetLength(I0);

            const bool has_main_k0_block_loop = GridwiseGemm::CalculateHasMainK0BlockLoop(K0);

            float ave_time = 0;

            if(has_main_k0_block_loop)
            {
                const auto kernel = kernel_gemm_xdlops_v2r5<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    CDataType,
                    remove_reference_t<DeviceConvFwdXdl_bias_activation_add::AGridDesc_K0_M_K1>,
                    remove_reference_t<DeviceConvFwdXdl_bias_activation_add::BGridDesc_K0_N_K1>,
                    remove_reference_t<
                        DeviceConvFwdXdl_bias_activation_add::CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2>,
                    remove_reference_t<
                        DeviceConvFwdXdl_bias_activation_add::C0GridDesc_M0_N0_M1_N1_M2_M3_M4_N2>,
                    remove_reference_t<
                        DeviceConvFwdXdl_bias_activation_add::C1GridDesc_M0_N0_M1_N1_M2_M3_M4_N2>,
                    InElementwiseOperation,
                    WeiElementwiseOperation,
                    OutElementwiseOperation,
                    remove_reference_t<DeviceConvFwdXdl_bias_activation_add::Block2CTileMap>,
                    true>;

                ave_time = launch_and_time_kernel(kernel,
                                                  nrepeat,
                                                  dim3(grid_size),
                                                  dim3(BlockSize),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg.p_c0_grid_,
                                                  arg.p_c1_grid_,
                                                  arg.a_grid_desc_k0_m_k1_,
                                                  arg.b_grid_desc_k0_n_k1_,
                                                  arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_,
                                                  arg.c0_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_,
                                                  arg.c1_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_,
                                                  arg.in_element_op_,
                                                  arg.wei_element_op_,
                                                  arg.out_element_op_,
                                                  arg.block_2_ctile_map_);
            }
            else
            {
                const auto kernel = kernel_gemm_xdlops_v2r5<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    CDataType,
                    remove_reference_t<DeviceConvFwdXdl_bias_activation_add::AGridDesc_K0_M_K1>,
                    remove_reference_t<DeviceConvFwdXdl_bias_activation_add::BGridDesc_K0_N_K1>,
                    remove_reference_t<
                        DeviceConvFwdXdl_bias_activation_add::CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2>,
                    remove_reference_t<
                        DeviceConvFwdXdl_bias_activation_add::C0GridDesc_M0_N0_M1_N1_M2_M3_M4_N2>,
                    remove_reference_t<
                        DeviceConvFwdXdl_bias_activation_add::C1GridDesc_M0_N0_M1_N1_M2_M3_M4_N2>,
                    InElementwiseOperation,
                    WeiElementwiseOperation,
                    OutElementwiseOperation,
                    remove_reference_t<DeviceConvFwdXdl_bias_activation_add::Block2CTileMap>,
                    false>;

                ave_time = launch_and_time_kernel(kernel,
                                                  nrepeat,
                                                  dim3(grid_size),
                                                  dim3(BlockSize),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg.p_c0_grid_,
                                                  arg.p_c1_grid_,
                                                  arg.a_grid_desc_k0_m_k1_,
                                                  arg.b_grid_desc_k0_n_k1_,
                                                  arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_,
                                                  arg.c0_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_,
                                                  arg.c1_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_,
                                                  arg.in_element_op_,
                                                  arg.wei_element_op_,
                                                  arg.out_element_op_,
                                                  arg.block_2_ctile_map_);
            }

            return ave_time;
        }

        // polymorphic
        float Run(const BaseArgument* p_arg, int nrepeat = 1) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), nrepeat);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        return GridwiseGemm::CheckValidity(arg.a_grid_desc_k0_m_k1_,
                                           arg.b_grid_desc_k0_n_k1_,
                                           arg.c_grid_desc_m_n_,
                                           arg.M01_,
                                           arg.N01_);
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const InDataType* p_in_grid,
                             const WeiDataType* p_wei_grid,
                             OutDataType* p_out_grid,
                             const OutDataType* p_bias_grid,
                             const OutDataType* p_resi_grid,
                             ck::index_t N,
                             ck::index_t K,
                             ck::index_t C,
                             std::vector<ck::index_t> input_spatial_lengths,
                             std::vector<ck::index_t> filter_spatial_lengths,
                             std::vector<ck::index_t> output_spatial_lengths,
                             std::vector<ck::index_t> conv_filter_strides,
                             std::vector<ck::index_t> conv_filter_dilations,
                             std::vector<ck::index_t> input_left_pads,
                             std::vector<ck::index_t> input_right_pads,
                             InElementwiseOperation in_element_op,
                             WeiElementwiseOperation wei_element_op,
                             OutElementwiseOperation out_element_op)
    {
        return Argument{p_in_grid,
                        p_wei_grid,
                        p_out_grid,
                        p_bias_grid,
                        p_resi_grid,
                        N,
                        K,
                        C,
                        input_spatial_lengths,
                        filter_spatial_lengths,
                        output_spatial_lengths,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        1,
                        1,
                        in_element_op,
                        wei_element_op,
                        out_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }
}; // namespace device

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
