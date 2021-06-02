#ifndef CK_DRIVER_DYNAMIC_GEMM_V1R2
#define CK_DRIVER_DYNAMIC_GEMM_V1R2

#include "common_header.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "gridwise_dynamic_gemm_v1r2.hpp"

namespace ck {

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          InMemoryDataOperation CGlobalMemoryDataOperation,
          typename AKMGridDesc,
          typename BKNGridDesc,
          typename CMNGridDesc,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t M1PerThread,
          index_t N1PerThread,
          index_t KPerThread,
          index_t M1N1ThreadClusterM10,
          index_t M1N1ThreadClusterN10,
          index_t M1N1ThreadClusterM11,
          index_t M1N1ThreadClusterN11,
          typename ABlockTransferThreadSliceLengths_K_M0_M1,
          typename ABlockTransferThreadClusterLengths_K_M0_M1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_M1,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          typename BBlockTransferThreadSliceLengths_K_N0_N1,
          typename BBlockTransferThreadClusterLengths_K_N0_N1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_N1,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          typename CThreadTransferSrcDstAccessOrder,
          index_t CThreadTransferSrcDstVectorDim,
          index_t CThreadTransferDstScalarPerVector,
          typename AGridIteratorHacks,
          typename BGridIteratorHacks,
          typename CGridIteratorHacks,
          typename AGridMoveSliceWindowIteratorHacks,
          typename BGridMoveSliceWindowIteratorHacks>
__host__ float driver_dynamic_gemm_v1r2(const FloatAB* p_a_grid,
                                        const FloatAB* p_b_grid,
                                        FloatC* p_c_grid,
                                        const AKMGridDesc& a_k_m_grid_desc,
                                        const BKNGridDesc& b_k_n_grid_desc,
                                        const CMNGridDesc& c_m_n_grid_desc,
                                        AGridIteratorHacks,
                                        BGridIteratorHacks,
                                        CGridIteratorHacks,
                                        AGridMoveSliceWindowIteratorHacks,
                                        BGridMoveSliceWindowIteratorHacks,
                                        index_t nrepeat)

{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

    // GEMM
    using GridwiseGemm =
        GridwiseDynamicGemm_km_kn_mn_v1r2<BlockSize,
                                          FloatAB,
                                          FloatAcc,
                                          FloatC,
                                          CGlobalMemoryDataOperation,
                                          AKMGridDesc,
                                          BKNGridDesc,
                                          CMNGridDesc,
                                          MPerBlock,
                                          NPerBlock,
                                          KPerBlock,
                                          M1PerThread,
                                          N1PerThread,
                                          KPerThread,
                                          M1N1ThreadClusterM10,
                                          M1N1ThreadClusterN10,
                                          M1N1ThreadClusterM11,
                                          M1N1ThreadClusterN11,
                                          ABlockTransferThreadSliceLengths_K_M0_M1,
                                          ABlockTransferThreadClusterLengths_K_M0_M1,
                                          ABlockTransferThreadClusterArrangeOrder,
                                          ABlockTransferSrcAccessOrder,
                                          ABlockTransferSrcVectorDim,
                                          ABlockTransferSrcScalarPerVector,
                                          ABlockTransferDstScalarPerVector_M1,
                                          AThreadTransferSrcResetCoordinateAfterRun,
                                          BBlockTransferThreadSliceLengths_K_N0_N1,
                                          BBlockTransferThreadClusterLengths_K_N0_N1,
                                          BBlockTransferThreadClusterArrangeOrder,
                                          BBlockTransferSrcAccessOrder,
                                          BBlockTransferSrcVectorDim,
                                          BBlockTransferSrcScalarPerVector,
                                          BBlockTransferDstScalarPerVector_N1,
                                          BThreadTransferSrcResetCoordinateAfterRun,
                                          CThreadTransferSrcDstAccessOrder,
                                          CThreadTransferSrcDstVectorDim,
                                          CThreadTransferDstScalarPerVector,
                                          AGridIteratorHacks,
                                          BGridIteratorHacks,
                                          CGridIteratorHacks,
                                          AGridMoveSliceWindowIteratorHacks,
                                          BGridMoveSliceWindowIteratorHacks>;

    const auto M = a_k_m_grid_desc.GetLength(I1);
    const auto N = b_k_n_grid_desc.GetLength(I1);
    const auto K = a_k_m_grid_desc.GetLength(I0);

    if(!GridwiseGemm::CheckValidity(a_k_m_grid_desc, b_k_n_grid_desc, c_m_n_grid_desc))
    {
        throw std::runtime_error("wrong! GridwiseDynamicGemm_km_kn_mn_v1r2 has invalid setting");
    }

    const auto a_k_m0_m1_grid_desc = GridwiseGemm::MakeAKM0M1GridDescriptor(a_k_m_grid_desc);
    const auto b_k_n0_n1_grid_desc = GridwiseGemm::MakeBKN0N1GridDescriptor(b_k_n_grid_desc);

    using AKM0M1GridDesc = decltype(a_k_m0_m1_grid_desc);
    using BKN0N1GridDesc = decltype(b_k_n0_n1_grid_desc);

    // c_m0_m10_m11_n0_n10_n11_grid_desc
    const auto c_m0_m10_m11_n0_n10_n11_grid_desc =
        GridwiseGemm::MakeCM0M10M11N0N10N11GridDescriptor(c_m_n_grid_desc);

    using CM0M10M11N0N10N11GridDesc = decltype(c_m0_m10_m11_n0_n10_n11_grid_desc);

    // c_blockid_to_m0_n0_block_cluster_adaptor
    const auto c_blockid_to_m0_n0_block_cluster_adaptor =
        GridwiseGemm::MakeCBlockIdToM0N0BlockClusterAdaptor(c_m_n_grid_desc);

    using CBlockIdToM0N0BlockClusterAdaptor = decltype(c_blockid_to_m0_n0_block_cluster_adaptor);

    const index_t grid_size = GridwiseGemm::CalculateGridSize(M, N);

    const bool has_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K);

    const bool has_double_tail_k_block_loop = GridwiseGemm::CalculateHasDoubleTailKBlockLoop(K);

    float ave_time = 0;

    if(has_main_k_block_loop && has_double_tail_k_block_loop)
    {
        const auto kernel =
            kernel_dynamic_gemm_v1r2<GridwiseGemm,
                                     FloatAB,
                                     FloatC,
                                     remove_reference_t<AKM0M1GridDesc>,
                                     remove_reference_t<BKN0N1GridDesc>,
                                     remove_reference_t<CM0M10M11N0N10N11GridDesc>,
                                     remove_reference_t<CBlockIdToM0N0BlockClusterAdaptor>,
                                     true,
                                     true>;

        ave_time = launch_and_time_kernel(kernel,
                                          nrepeat,
                                          dim3(grid_size),
                                          dim3(BlockSize),
                                          0,
                                          0,
                                          p_a_grid,
                                          p_b_grid,
                                          p_c_grid,
                                          a_k_m0_m1_grid_desc,
                                          b_k_n0_n1_grid_desc,
                                          c_m0_m10_m11_n0_n10_n11_grid_desc,
                                          c_blockid_to_m0_n0_block_cluster_adaptor);
    }
    else if(has_main_k_block_loop && !has_double_tail_k_block_loop)
    {
        const auto kernel =
            kernel_dynamic_gemm_v1r2<GridwiseGemm,
                                     FloatAB,
                                     FloatC,
                                     remove_reference_t<AKM0M1GridDesc>,
                                     remove_reference_t<BKN0N1GridDesc>,
                                     remove_reference_t<CM0M10M11N0N10N11GridDesc>,
                                     remove_reference_t<CBlockIdToM0N0BlockClusterAdaptor>,
                                     true,
                                     false>;

        ave_time = launch_and_time_kernel(kernel,
                                          nrepeat,
                                          dim3(grid_size),
                                          dim3(BlockSize),
                                          0,
                                          0,
                                          p_a_grid,
                                          p_b_grid,
                                          p_c_grid,
                                          a_k_m0_m1_grid_desc,
                                          b_k_n0_n1_grid_desc,
                                          c_m0_m10_m11_n0_n10_n11_grid_desc,
                                          c_blockid_to_m0_n0_block_cluster_adaptor);
    }
    else if(!has_main_k_block_loop && has_double_tail_k_block_loop)
    {
        const auto kernel =
            kernel_dynamic_gemm_v1r2<GridwiseGemm,
                                     FloatAB,
                                     FloatC,
                                     remove_reference_t<AKM0M1GridDesc>,
                                     remove_reference_t<BKN0N1GridDesc>,
                                     remove_reference_t<CM0M10M11N0N10N11GridDesc>,
                                     remove_reference_t<CBlockIdToM0N0BlockClusterAdaptor>,
                                     false,
                                     true>;

        ave_time = launch_and_time_kernel(kernel,
                                          nrepeat,
                                          dim3(grid_size),
                                          dim3(BlockSize),
                                          0,
                                          0,
                                          p_a_grid,
                                          p_b_grid,
                                          p_c_grid,
                                          a_k_m0_m1_grid_desc,
                                          b_k_n0_n1_grid_desc,
                                          c_m0_m10_m11_n0_n10_n11_grid_desc,
                                          c_blockid_to_m0_n0_block_cluster_adaptor);
    }
    else
    {
        const auto kernel =
            kernel_dynamic_gemm_v1r2<GridwiseGemm,
                                     FloatAB,
                                     FloatC,
                                     remove_reference_t<AKM0M1GridDesc>,
                                     remove_reference_t<BKN0N1GridDesc>,
                                     remove_reference_t<CM0M10M11N0N10N11GridDesc>,
                                     remove_reference_t<CBlockIdToM0N0BlockClusterAdaptor>,
                                     false,
                                     false>;

        ave_time = launch_and_time_kernel(kernel,
                                          nrepeat,
                                          dim3(grid_size),
                                          dim3(BlockSize),
                                          0,
                                          0,
                                          p_a_grid,
                                          p_b_grid,
                                          p_c_grid,
                                          a_k_m0_m1_grid_desc,
                                          b_k_n0_n1_grid_desc,
                                          c_m0_m10_m11_n0_n10_n11_grid_desc,
                                          c_blockid_to_m0_n0_block_cluster_adaptor);
    }

    return ave_time;
}

} // namespace ck
#endif
